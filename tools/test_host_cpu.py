#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""CPU-only production host buffer/writer and extracted daemon ingress tests."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
COMMON=r'''
#include "hive/host_image.h"
#include "hive/socket_writer.h"
#include "nlohmann/json.hpp"
#include <fcntl.h>
#include <sys/un.h>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <memory>
using namespace hive;
using json=nlohmann::json;
SocketWriter* output;
bool send_json(int fd,const json& j){return output->send(fd,j.dump()+"\n");}
void finish_socket(int fd){output->finish(fd);}
struct Request {int fd; json req; std::vector<uint8_t> bin; std::shared_ptr<std::atomic<bool>> cancel; double t_recv=0;};  // t_recv: for queue_ms logging
'''
TEST=r'''
int main(){
 auto alloc=[](size_t n){return std::shared_ptr<uint8_t>(new uint8_t[n],std::default_delete<uint8_t[]>());};
 HostImageBuffer a,b,c;
 assert(a.extend(nullptr,100,alloc)==0); a.segments[0].data.get()[0]=42;
 assert(b.extend(&a,150,alloc)==100); assert(b.segments.size()==2);
 assert(c.extend(&a,80,alloc)==0); assert(c.segments[0].data!=a.segments[0].data);
 std::set<const void*> seen;
 size_t bytes=a.allocated_bytes(seen)+b.allocated_bytes(seen);
 assert(bytes==150+(a.segments.capacity()+b.segments.capacity())*sizeof(HostImageBuffer::Segment));
 a.segments.clear(); assert(b.segments[0].data.get()[0]==42);
 int allocations=0,frees=0;
 auto pool=std::make_shared<HostImagePool>(128,[&](size_t n){++allocations;return new uint8_t[n];},[&](uint8_t* p){++frees;delete[] p;});
 {auto x=pool->acquire(64);}assert(pool->cached_bytes()==64);
 {auto x=pool->acquire(64);assert(allocations==1);auto y=pool->acquire(129);}assert(frees==1);
 auto survivor=pool->acquire(64);pool.reset();assert(frees==1);survivor.reset();assert(frees==2);
 int slow[2],fast[2]; assert(socketpair(AF_UNIX,SOCK_STREAM,0,slow)==0); assert(socketpair(AF_UNIX,SOCK_STREAM,0,fast)==0);
 int small=4096; setsockopt(slow[0],SOL_SOCKET,SO_SNDBUF,&small,sizeof(small));
 SocketWriter w; w.add(slow[0]);w.add(fast[0]);
 // Would block the former engine-thread write_all before reaching the fast peer.
 assert(w.send(slow[0],std::string(4*1024*1024,'x')));
 assert(w.send(fast[0],"ready\n")); w.finish(fast[0]);
 pollfd p{fast[1],POLLIN,0}; assert(poll(&p,1,1000)>0);
 char buf[16]{}; assert(read(fast[1],buf,16)==6); assert(std::string(buf)=="ready\n");
 close(fast[1]);close(slow[1]);w.finish(slow[0]);
 int bounded[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,bounded)==0);
 SocketWriter limited(64,30);limited.add(bounded[0]);
 assert(!limited.send(bounded[0],std::string(65,'x')));limited.finish(bounded[0]);
 pollfd closed{bounded[1],POLLIN,0};assert(poll(&closed,1,1000)>0);assert(read(bounded[1],buf,sizeof(buf))==0);close(bounded[1]);
 int abandoned[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,abandoned)==0);
 setsockopt(abandoned[0],SOL_SOCKET,SO_SNDBUF,&small,sizeof(small));
 SocketWriter drain(8u<<20,30);drain.add(abandoned[0]);assert(drain.send(abandoned[0],std::string(4*1024*1024,'x')));drain.finish(abandoned[0]);
 std::this_thread::sleep_for(std::chrono::milliseconds(100));
 pollfd ended{abandoned[1],POLLIN,0};assert(poll(&ended,1,1000)>0);assert(ended.revents&POLLHUP);close(abandoned[1]);
 // D10: a vanished peer is reported by send() itself (inline non-blocking write on an empty queue), not one send later
 int gone[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,gone)==0);SocketWriter inl;inl.add(gone[0]);
 assert(inl.send(gone[0],"a\n"));char two[4]{};assert(read(gone[1],two,4)==2);  // delivered inline, nothing queued
 close(gone[1]);assert(!inl.send(gone[0],"b\n"));assert(!inl.send(gone[0],"c\n"));inl.finish(gone[0]);
 // partial inline write keeps order: head goes inline, the rest is queued behind it and then later sends
 int part[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,part)==0);setsockopt(part[0],SOL_SOCKET,SO_SNDBUF,&small,sizeof(small));
 SocketWriter ord;ord.add(part[0]);std::string big;for(int i=0;i<200000;++i) big+=(char)('a'+i%26);
 assert(ord.send(part[0],big));assert(ord.send(part[0],"END"));ord.finish(part[0]);
 std::string got;char rb[8192];for(;;){ssize_t n=read(part[1],rb,sizeof(rb));if(n<=0)break;got.append(rb,(size_t)n);}
 assert(got==big+"END");close(part[1]);
 puts("host CPU: incremental sharing/branch/accounting, slow-reader isolation, ordered drain, inline EPIPE/partial order OK");
}
'''
src=(ROOT/'engine/src/hived.cpp').read_text()
runtime_header=(ROOT/'engine/include/hive/runtime.h').read_text()
image_types=runtime_header[runtime_header.index('struct Seq {'):runtime_header.index('struct RuntimeOptions {')]
signature_check=src[src.index('    auto images_match ='):src.index('    // Pick where to continue writing')]
LIFETIME=r'''
struct SnapshotFence {int waits=0;void wait(){++waits;}};
struct DevBuf {};
'''+image_types+r'''
int main(){
 auto fence=std::make_shared<SnapshotFence>();
 {SeqImage image;image.fence=fence;}
 assert(fence->waits==1); // another shared owner must NOT defer the wait
 {Seq seq;seq.snapshot_fence=fence;}
 assert(fence->waits==2);
 struct Session {struct ImgSig{int start,end;uint64_t hash;};};
 std::vector<Session::ImgSig> request_sig{{10,20,1}};
'''+signature_check+r'''
 assert(images_match({{10,20,1}},20));assert(!images_match({{10,20,2}},20));
 assert(!images_match({},20));assert(!images_match({{10,20,1}},15));
 assert(images_match({},10));request_sig.clear();assert(!images_match({{10,20,1}},20));
 puts("snapshot CPU: shared-owner destruction fences and exact image-prefix signatures OK");
}
'''
recovery=src[src.index('  auto decode_inner ='):src.index('  // ---- sleep / wake (engine thread)')]  # the Z1 block (sleep state machine) stays outside this slice
# the [step-host] state the wrapper reads (HIVE_PROFILE) is declared right before decode_step's body — taken from hived.cpp as well
step_host_state=src[src.index('  struct StepHost {'):src.index('  decode_step = [&] {')]
RECOVERY=r'''
#include "hive/clock.h"
double now_ms() { return hive::mono_ms(); }
int main(){
 long prefill_epoch=0;
'''+step_host_state+r'''
 struct Sequence {bool broken=false;};struct Session {std::unique_ptr<Sequence> seq=std::make_unique<Sequence>();} session;
 struct Active {Session* S;Request r;std::string sid;std::atomic<bool>* cancel;bool client_ok=true;};
 struct Runtime {int count=0;void quiesce_after_host_error(){++count;}} rt;
 SocketWriter writer;output=&writer;int fd[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0);writer.add(fd[0]);
 std::mutex mu;std::map<std::string,std::shared_ptr<std::atomic<bool>>> cancel_flags;
 cancel_flags["test"]=std::make_shared<std::atomic<bool>>(false);
 std::set<std::string> active_sids{"test"};std::vector<std::unique_ptr<Active>> active;
 auto a=std::make_unique<Active>();a->S=&session;a->r.fd=fd[0];a->sid="test";a->cancel=cancel_flags["test"].get();active.push_back(std::move(a));
 std::function<void()> decode_step=[](){throw std::runtime_error("injected host failure");};
'''+recovery+r'''
 decode_step();assert(active.empty() && active_sids.empty() && cancel_flags.empty());assert(session.seq->broken && rt.count==1);
 pollfd p{fd[1],POLLIN,0};assert(poll(&p,1,1000)>0);char buf[256]{};assert(read(fd[1],buf,255)>0);assert(std::string(buf).find("decode failed")!=std::string::npos);close(fd[1]);
 puts("decode CPU: injected host exception, quiescence boundary, session/fd/cancel cleanup OK");
}
'''
ingress=src[src.index('  std::thread acceptor([&] {'):src.index('  // ---- processing loop')]
SERVER=r'''
int main(int argc,char** argv){
 int lfd=socket(AF_UNIX,SOCK_STREAM,0);sockaddr_un a{};a.sun_family=AF_UNIX;strncpy(a.sun_path,argv[1],sizeof(a.sun_path)-1);
 assert(bind(lfd,(sockaddr*)&a,sizeof(a))==0);assert(listen(lfd,16)==0);
 SocketWriter writer;output=&writer;
 std::mutex mu,stats_mu;std::condition_variable cv;std::deque<Request> queue;
 std::map<std::string,std::shared_ptr<std::atomic<bool>>> cancel_flags;std::map<std::string,std::weak_ptr<std::atomic<bool>>> rid_flags;
 json stats_snapshot={{"build_id","cpu-fixture"}};
 // graceful stop state (hived HIVE_GRACEFUL_STOP_S): off in this fixture, as in the default daemon
 std::atomic<int> stop_state{0};int stop_pipe[2]={-1,-1};const double graceful_stop_s=0;
 // Z1 sleep/wake control ops (hived {"op":"sleep"/"wake"}): queued for the engine thread — none in this fixture
 struct Ctl{int fd;bool sleep;double deadline;int level;};std::deque<Ctl> ctl_q;std::atomic<bool> sleep_asked{false};
 std::vector<int> flush_q;  // {"op":"flush"} connections (hived) — none are drained in this fixture
 std::atomic<int> g_mtp_batch_override{-1};  // {"op":"set","mtp_batch":...} run-time override (hived global — the engine thread is not part of this fixture)
 const size_t kMaxHeaderBytes=(size_t)64<<20,kMaxBinBytes=(size_t)1<<30;  // request size caps (hived, next to the listening socket)
 auto now_ms=[]{return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();};
'''+ingress+r'''
 acceptor.join();
}
'''
WAIT_IDLE=r'''
int main(){
 // SocketWriter::wait_idle (hived graceful stop): true once every finished peer's bytes are sent and the fd closed; false on timeout
 SocketWriter w;int fd[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0);w.add(fd[0]);
 assert(w.send(fd[0],"terminal\n"));assert(!w.wait_idle(50));  // not finished yet: stays
 w.finish(fd[0]);assert(w.wait_idle(2000));
 char buf[64]{};assert(read(fd[1],buf,63)==9 && std::string(buf)=="terminal\n");assert(read(fd[1],buf,63)==0);close(fd[1]);
 puts("socket writer CPU: wait_idle flushes finished peers OK");
}
'''
with tempfile.TemporaryDirectory(prefix='hive-host-test-') as td:
    def build(name, source):
        exe=str(Path(td)/name)
        subprocess.run(['g++','-std=c++20','-O1','-g','-pthread','-fsanitize='+os.environ.get('HIVE_TEST_SANITIZER','undefined'),'-fno-sanitize-recover=all',
                        '-I'+str(ROOT/'engine/include'),'-I'+str(ROOT/'engine/third_party'),
                        '-x','c++','-','-o',exe],input=COMMON+source,text=True,check=True)
        return exe
    subprocess.run([build('host',TEST)],check=True,timeout=15)
    subprocess.run([build('lifetime',LIFETIME)],check=True,timeout=15)
    subprocess.run([build('recovery',RECOVERY)],check=True,timeout=15)
    subprocess.run([build('wait_idle',WAIT_IDLE)],check=True,timeout=15)
    path=str(Path(td)/'test.sock')
    proc=subprocess.Popen([build('ingress',SERVER),path])
    sockets=[]
    try:
        deadline=time.monotonic()+5
        while not Path(path).exists():
            if proc.poll() is not None or time.monotonic()>deadline: raise RuntimeError('ingress failed to start')
            time.sleep(.01)
        def connect(payload):
            sock=socket.socket(socket.AF_UNIX);sock.settimeout(1);sock.connect(path);sock.sendall(payload);sockets.append(sock);return sock
        connect(b'{"op":')  # abandoned header
        connect(b'{"op":"generate","bin":1000000}\npartial-image')
        result=json.loads(connect(b'{"op":"stats"}\n').recv(4096))
        assert result['build_id']=='cpu-fixture'
        result=json.loads(connect(b'{"op":"cancel","session":42}\n').recv(4096))
        assert result['ok']
        result=json.loads(connect(b'not-json\n').recv(4096))
        assert 'error' in result
        assert json.loads(connect(b'{"op":"stats"}\n').recv(4096))['build_id']=='cpu-fixture'
        print('ingress CPU: partial header/image isolation, malformed envelope, cancel/stats survival OK')
    finally:
        for sock in sockets: sock.close()
        proc.terminate();proc.wait(timeout=5)
