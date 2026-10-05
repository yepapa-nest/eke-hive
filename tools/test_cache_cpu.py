#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Compile production cache methods against deterministic fake CUDA events (no GPU)."""
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
src = (ROOT / 'engine/src/expert_store.cpp').read_text()
code = r'''
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <tuple>
#include <vector>
using cudaStream_t=int; using cudaEvent_t=int;
const int cudaSuccess=0,cudaErrorNotReady=1,cudaEventDisableTiming=0;
int event_status=cudaErrorNotReady;
int cudaEventCreateWithFlags(int* e,int){static int n=0;*e=++n;return 0;}
// (op, event, stream): 'r' record, 'w' wait — the stream matters (REUSE_STAGE D2D has its own stream 9)
using Ev=std::tuple<char,int,int>;
std::vector<Ev> event_log;
// E4: one ordered log of every stream op (record/wait/memcpy/sync) for the HIVE_DECODE_COPY_PRIO pacing checks — (op, a, stream, bytes)
struct Op {char op;int a,s;size_t n;bool operator==(const Op&)const=default;};
std::vector<Op> ops;
int cudaEventRecord(int e,int s){event_log.push_back({'r',e,s});ops.push_back({'r',e,s,0});return 0;}
int cudaStreamWaitEvent(int s,int e,int){event_log.push_back({'w',e,s});ops.push_back({'w',e,s,0});return 0;}
std::vector<int> d2d_streams;
const int cudaMemcpyDeviceToDevice=1,cudaMemcpyHostToDevice=2;
int cudaMemcpyAsync(void*,const void*,size_t n,int kind,int s){d2d_streams.push_back(s);ops.push_back({'m',kind,s,n});return 0;}
int cudaEventQuery(int){return event_status;}
int cudaEventSynchronize(int e){event_status=0;ops.push_back({'S',e,0,0});return 0;}
#define CUDA_CHECK(x) do {if ((x)!=0) throw std::runtime_error("CUDA failure");} while(0)
#define HIVE_CHECK(x,msg) do {if (!(x)) throw std::runtime_error(msg);} while(0)
struct ExpertStore {
 int n_slots_=4,E_=8; uint64_t use_total_=0;
 std::vector<int> slot_of_=std::vector<int>(8,-1),expert_of_slot_=std::vector<int>(4,-1),pending_=std::vector<int>(4,-1),pending_slot_=std::vector<int>(8,-1);
 std::vector<float> score_=std::vector<float>(8,0);
 // D4 HIVE_CACHE_POLICY members referenced by the sliced promote/seed/resident code (policy off here = the default path)
 int policy_=0; std::vector<float> prio_; float p_min_=0.025f;
 float prio_of(int k)const{return policy_?prio_[k]:score_[k];}
 std::vector<uint64_t> slot_last_use_=std::vector<uint64_t>(4,0);
 uint64_t promotions_=0,commits_=0,evictions_=0,duplicate_skips_=0,h2d_records_=0,d2d_records_=0;
 struct CacheStats {int resident=0,unique=0,pending=0,duplicates=0,invalid_mappings=0;uint64_t promotions=0,commits=0,evictions=0,duplicate_skips=0,h2d_records=0,d2d_records=0;};
 struct PromoBatch {int evt;std::vector<int> slots;bool paced=false,issued=true;int pieces_left=0;int st=0;};
 double commit_wait_ms_=0;  // commit_pending: host time waited for paced batches
 // E4 HIVE_DECODE_COPY_PRIO members (same names as expert_store.h); mock record = 12 pieces {5,1,5,1,5,1}×2 bytes into a real buffer
 bool pace_=false; struct PromoPiece{uint8_t* dst;const uint8_t* src;size_t n;}; std::deque<PromoPiece> pace_pieces_;
 size_t pace_bytes_=0; int pace_unissued_=0,pace_batch_pieces_=0; uint64_t promo_h2d_bytes_=0;
 uint8_t dev_[4*64]={}; uint8_t host_[64]={};
 template<class F> void for_each_rec_piece(uint8_t* d,int,int,F&& f) const {static const size_t sz[6]={5,1,5,1,5,1};size_t off=0;
  for(int n=0;n<2;++n) for(int i=0;i<6;++i){f(d+off,host_+off,sz[i]);off+=sz[i];}}
 void defer_rec(uint8_t*,int,int);size_t issue_promotions(size_t);
 void flush_promotions(){if(pace_unissued_>0) issue_promotions(SIZE_MAX);}
 bool promo_backlog()const{return pace_unissued_>0;} size_t promo_backlog_bytes()const{return pace_bytes_;}
 std::deque<PromoBatch> promo_q_;std::vector<int> evt_pool_;
 int E_of(int)const{return E_;}
 int n_pending()const{return std::count_if(pending_.begin(),pending_.end(),[](int x){return x>=0;});}
 const uint8_t* dev_rec(int s){return dev_+s*64;}
 void copy_rec_async(uint8_t*,int,int,int){++h2d_records_;}
 bool reuse_staging_=false;int n_staging_=2;
 struct Layout {size_t total=16;} lay_;
 std::vector<int> staging_key_{-1,-1},staging_ready_{101,102},staging_reused_{103,104},staging_reuse_pending_{0,0};
 int reuse_st_=9,reuse_done_=105;bool reuse_join_pending_=false;
 uint8_t* staging_rec(int){return nullptr;}
 void copy_for_promotion(uint8_t*,int,int,int,int=0);void copy_to_staging(int,int,int,int);void join_reuse(int);
 int place(int,int,int);int promote(int,float,int,int=0);
 int promote_keys(const std::vector<int>&,int,int,int,bool=false,int=0);
 void commit_pending();void commit_all();
 std::vector<int32_t> resident_keys_by_score()const;
 void seed_scores(const std::vector<int32_t>&,float,float);
 CacheStats cache_stats()const;
 void trace_event(char,int,int)const{}
};
'''
code += src[src.index('int ExpertStore::place('):src.index('std::vector<std::pair<int, double>> ExpertStore::coverage(')]
code += src[src.index('void ExpertStore::defer_rec('):src.index('void ExpertStore::copy_to_staging(')]
code += src[src.index('void ExpertStore::copy_to_staging('):src.index('void ExpertStore::load_layer_experts(')]
code += src[src.index('void ExpertStore::commit_all('):src.index('void ExpertStore::run_jobs(')]
rt = (ROOT / 'engine/src/runtime.cpp').read_text()
pump = rt[rt.index('void Runtime::promo_pump(int l, int nL) {'):]
pump = pump[:pump.index('\n}\n') + 3].replace('Runtime::', 'RtShell::')
code += r'''
struct RtShell {bool copy_prio_=true;int promo_gate_=201,side_=1,promo_=2;ExpertStore& store_;void promo_pump(int l,int nL);};
''' + pump
code += r'''
void valid(const ExpertStore& s){auto m=s.cache_stats();assert(m.duplicates==0);assert(m.invalid_mappings==0);assert(m.resident+m.pending<=4);}
int main(){
 {ExpertStore s;s.score_[0]=100;event_status=1;
 assert(s.promote(1,2,0)==1);assert(s.promote(1,2,0)==0);
 assert(s.promote_keys({0,0},1,0,1)==0);valid(s);
 event_status=0;s.commit_pending();assert(s.slot_of_[0]>=0);valid(s);
 s.expert_of_slot_[1]=1;s.slot_of_[1]=1;s.score_[1]=200;
 s.expert_of_slot_[2]=2;s.slot_of_[2]=2;s.score_[2]=200;
 s.expert_of_slot_[3]=3;s.slot_of_[3]=3;s.score_[3]=200;
 s.score_[4]=300;assert(s.promote(1,2,0)==1);s.commit_pending();valid(s);
 assert(s.slot_of_[0]==-1);assert(s.slot_of_[4]>=0);}
 {ExpertStore s;s.score_[0]=10;event_status=1;s.promote(1,2,0);
 assert(s.place(0,0,0)==0);assert(s.h2d_records_==1);valid(s);}
 {ExpertStore s;s.seed_scores({0,0,1,7,99,-1},10,1);assert(s.score_[0]==10 && s.score_[1]==9 && s.score_[7]==8);}
 {ExpertStore s;s.score_[0]=10;event_status=1;s.promote(1,2,0);event_status=2;
 bool threw=false;try{s.commit_pending();}catch(const std::runtime_error&){threw=true;}assert(threw);}
 {ExpertStore s;s.score_[0]=10;event_status=1;s.promote(1,2,0);
 s.expert_of_slot_[1]=0;s.slot_of_[0]=1;event_status=0;s.commit_pending();valid(s);
 assert(s.slot_of_[0]==1);assert(s.resident_keys_by_score().size()==1);}
 {ExpertStore s;event_status=0;
 for(int k=0;k<8;++k){s.score_[k]=k+2;s.promote_keys({k,k},1,0,1);s.commit_pending();valid(s);}
 assert(s.cache_stats().resident==4);}
 {ExpertStore s;s.reuse_staging_=true;event_log.clear();d2d_streams.clear();
 s.copy_to_staging(0,0,1,1);s.copy_for_promotion(nullptr,0,1,2);
 assert(s.h2d_records_==1 && s.d2d_records_==1 && s.staging_reuse_pending_[0] && s.reuse_join_pending_);
 // D2D + reuse event on the dedicated stream 9, not behind the promotion stream 2's H2D queue
 assert((event_log==std::vector<Ev>{{'r',101,1},{'w',101,9},{'r',103,9}}));
 assert((d2d_streams==std::vector<int>{9}));
 s.copy_to_staging(0,0,2,1);assert((event_log[3]==Ev{'w',103,1}));
 assert(!s.staging_reuse_pending_[0] && s.staging_key_[0]==2);
 s.copy_for_promotion(nullptr,0,1,2);assert(s.h2d_records_==3);
 s.join_reuse(2);assert((event_log.back()==Ev{'w',105,2}) && !s.reuse_join_pending_);
 const size_t n=event_log.size();s.join_reuse(2);assert(event_log.size()==n);}
 // event order through promote_keys: victim fence -> staging ready -> D2D -> reuse event (stream 9), join (9 -> 2), batch event on 2 last
 {ExpertStore s;s.reuse_staging_=true;event_status=1;event_log.clear();d2d_streams.clear();
 s.copy_to_staging(1,0,5,1);event_log.clear();
 s.score_[5]=50;assert(s.promote_keys({5},1,2,1,false,77)==1);
 const int batch=s.promo_q_.back().evt;
 assert((event_log==std::vector<Ev>{{'w',77,9},{'w',102,9},{'r',104,9},{'r',105,9},{'w',105,2},{'r',batch,2}}));
 assert((d2d_streams==std::vector<int>{9}) && s.d2d_records_==1 && !s.reuse_join_pending_);
 // not staged -> plain H2D on the promotion stream, no D2D stream traffic, no join
 event_status=0;s.commit_pending();event_status=1;event_log.clear();
 s.score_[6]=60;assert(s.promote_keys({6},1,2,1,false,77)==1);
 assert((event_log==std::vector<Ev>{{'r',s.promo_q_.back().evt,2}}));
 // promote() (score path) takes the same route
 event_status=0;s.commit_pending();event_status=1;event_log.clear();
 s.copy_to_staging(0,0,7,1);event_log.clear();s.score_[7]=500;
 assert(s.promote(1,2,2,78)==1);
 assert((event_log==std::vector<Ev>{{'w',78,9},{'w',101,9},{'r',103,9},{'r',105,9},{'w',105,2},{'r',s.promo_q_.back().evt,2}}));
 event_status=0;s.commit_all();valid(s);}
 {ExpertStore s;event_status=0;for(int k=0;k<4;++k){s.place(0,k,0);s.score_[k]=10+k;}
 s.score_[4]=1;s.score_[5]=100;assert(s.promote_keys({4,5},1,0,1,true)==1);s.commit_pending();
 assert(s.slot_of_[0]==-1 && s.slot_of_[5]>=0);valid(s);}
 // ---- E4 HIVE_DECODE_COPY_PRIO: deferred promotion H2D, paced behind each layer's demand copies ----
 {ExpertStore s;s.pace_=true;event_status=0;ops.clear();
 assert(s.promote_keys({0,1},2,2,2)==1);  // n_slots/4 = 1 per batch (same cap as off)
 // decision = same as off (pending marked, h2d counted, promotions++) but no copy and no batch event yet
 assert(s.n_pending()==1 && s.h2d_records_==1 && s.promotions_==1 && ops.empty());
 assert(s.promo_backlog() && s.promo_backlog_bytes()==36 && s.pace_pieces_.size()==12 && !s.promo_q_.back().issued);
 // an unrecorded event answers "done" (fake + real CUDA) — commit_pending must not commit an unissued batch
 s.commit_pending();assert(s.n_pending()==1 && s.slot_of_[0]<0);
 const int batch=s.promo_q_.back().evt;
 RtShell rt{true,201,1,2,s};const int nL=5;size_t prev_left=s.promo_backlog_bytes();
 for(int l=0;l<nL;++l){
  ops.push_back({'D',l,1,0});  // this layer's demand DMA on side_ (stream 1)
  const size_t left=s.promo_backlog_bytes(),budget=(left+(nL-l)-1)/(nL-l);
  const size_t o0=ops.size();rt.promo_pump(l,nL);
  // gate on side_ after the demand copies, promo_ waits for it, then pieces on promo_ only
  assert((ops[o0]==Op{'r',201,1,0}) && (ops[o0+1]==Op{'w',201,2,0}));
  size_t sent=0;for(size_t i=o0+2;i<ops.size();++i){if(ops[i].op=='m'){assert(ops[i].s==2 && ops[i].a==2);sent+=ops[i].n;}else assert((ops[i]==Op{'r',batch,2,0}) && i+1==ops.size());}
  assert(sent>0 && sent<budget+5 && sent==prev_left-s.promo_backlog_bytes());prev_left=s.promo_backlog_bytes();}
 // every piece issued by the last layer; batch event recorded right after its last piece; same bytes as the immediate path (36)
 assert(!s.promo_backlog() && s.promo_backlog_bytes()==0 && s.promo_h2d_bytes_==36 && s.promo_q_.back().issued);
 assert((ops.back()==Op{'r',batch,2,0}));
 // head of the next step: waits for the batch (sync), commits in FIFO order
 ops.clear();s.commit_pending();assert((ops.front()==Op{'S',batch,0,0}) && s.slot_of_[0]>=0 && s.n_pending()==0);valid(s);
 // off switch: no pacing work at all
 RtShell off{false,201,1,2,s};ops.clear();off.promo_pump(0,nL);assert(ops.empty());}
 {// FIFO + flush at the next decision + zero-piece (REUSE_STAGE D2D-only) batch + commit_all flush
 ExpertStore s;s.pace_=true;event_status=0;ops.clear();
 assert(s.promote_keys({2},1,2,1)==1);const int b1=s.promo_q_.back().evt;
 assert(s.issue_promotions(0)==5 && s.promo_backlog());            // budget 0 still sends one piece (progress)
 assert(s.promote_keys({3},1,2,1)==1);const int b2=s.promo_q_.back().evt;  // flushes b1's remaining pieces first, then defers b2
 {size_t i=0;while(!(ops[i]==Op{'r',b1,2,0}))++i;for(size_t j=i+1;j<ops.size();++j)assert(ops[j].op!='m');}
 assert(s.promo_q_.front().issued && !s.promo_q_.back().issued);
 s.commit_pending();assert(s.slot_of_[2]>=0 && s.slot_of_[3]<0 && s.promo_q_.size()==1);  // b1 resident, b2 (unissued) waits
 s.commit_all();assert(s.slot_of_[3]>=0 && s.promo_q_.empty() && !s.promo_backlog());valid(s);(void)b2;
 s.reuse_staging_=true;s.copy_to_staging(0,0,6,1);ops.clear();s.score_[6]=60;
 assert(s.promote_keys({6},1,2,1,false,77)==1);                      // D2D now on stream 9 (as off), no H2D pieces
 assert(s.pace_pieces_.empty() && s.promo_backlog() && s.promo_q_.back().pieces_left==0);
 const int b3=s.promo_q_.back().evt;ops.clear();assert(s.issue_promotions(1)==0);
 assert((ops==std::vector<Op>{{'r',b3,2,0}}) && !s.promo_backlog());s.commit_pending();assert(s.slot_of_[6]>=0);valid(s);}
 puts("cache CPU: HIVE_DECODE_COPY_PRIO deferred promotion H2D (same decisions, no event before the last piece, gate after demand DMA, per-layer budget, all issued by the last layer, FIFO, flush on next decision/commit_all, D2D-only batch) OK");
 puts("cache CPU: duplicate pending, mixed insert, place, eviction, restore, event error, legacy normalization, REUSE_STAGE dedicated D2D stream event order OK");
}
'''
with tempfile.TemporaryDirectory(prefix='hive-cache-test-') as td:
    exe = str(pathlib.Path(td) / 'test')
    subprocess.run(['g++', '-O1', '-g', '-std=c++20', '-fsanitize=undefined', '-fno-sanitize-recover=all', '-x', 'c++', '-', '-o', exe], input=code, text=True, check=True)
    subprocess.run([exe], check=True)
