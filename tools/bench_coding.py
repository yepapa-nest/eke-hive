#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Adapted from hyudryu/Qwen3.8-Flash-Next-FP4-1xRTXPro6000 V1/benchmarks/coding.py (MIT License, Copyright (c) 2026 hyudryu) —
#   the coding throughput workload behind that project's published RTX PRO 6000 numbers (C1/C4/C8, 1,536 output tokens, thinking off).
#   Changes for Eke Hive: no vLLM /metrics counter checks (the server has no such endpoint), default base URL / model, optional --ignore-eos
#   (equal work per request when a model stops before the limit), the run's salt inside the prompt nonce (the server keeps prompt prefixes
#   across runs and ignores cache_salt). Request body and aggregation are unchanged.
"""Single-GPU coding throughput: natural EOS, paired prompts, unique cache salts."""
import argparse,concurrent.futures,hashlib,json,pathlib,statistics,sys,threading,time,urllib.request,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
def request(base,path,body=None):
 headers={'Content-Type':'application/json'}
 import os
 if os.environ.get('BENCH_API_KEY'):headers['Authorization']='Bearer '+os.environ['BENCH_API_KEY']
 req=urllib.request.Request(base.rstrip('/')+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
 return urllib.request.urlopen(req,timeout=120)
def sse(response):
 data=[]
 for raw in response:
  line=raw.decode().rstrip('\r\n')
  if line.startswith('data:'):data.append(line[5:].lstrip())
  elif not line and data:
   value='\n'.join(data);data=[]
   if value=='[DONE]':return
   yield json.loads(value)
 if data and '\n'.join(data)!='[DONE]':yield json.loads('\n'.join(data))
TASKS=[
('Python','Implement a production-quality asyncio TTL cache with async get_or_load, per-key single-flight loading, bounded LRU eviction, injectable monotonic clock, cancellation-safe loading, exception propagation, invalidate, and clear. Use only the standard library. Include unittest.IsolatedAsyncioTestCase tests for concurrent callers, failed loaders, TTL expiry, cancellation and eviction.'),
('TypeScript','Implement a strongly typed bounded asynchronous work queue. Support concurrency limits, backpressure, AbortSignal cancellation, graceful drain, ordered result collection, and a close method. Avoid dependencies. Include executable tests using node:test and demonstrate concurrent producers. Carefully distinguish pending and running cancellation.'),
('Rust','Implement a bounded thread-safe LRU cache with TTL using only the Rust standard library. Support get, insert, remove, len, deterministic injectable clock for tests, and safe shared access with Arc and Mutex. Avoid unsafe code. Explain ownership choices briefly and include unit tests for expiry, replacement, eviction and concurrent access.'),
('Go','Implement a bounded worker pool with context cancellation, task submission with backpressure, panic recovery, graceful shutdown, and deterministic result delivery. Avoid goroutine leaks and data races. Use only the Go standard library. Include tests for concurrency bounds, cancellation, shutdown, panic propagation and rejected submissions.')]
GUIDANCE='''Return an actual implementation and substantive tests, not a design sketch. Start with code. State the public API clearly in comments and handle edge cases explicitly. Keep all code self-contained. Include imports, type definitions, constructors, helper methods and tests. Prefer understandable correct code to clever tricks. Use concise comments where concurrency invariants matter. Do not use placeholder implementations, ellipses, TODOs, pseudocode, or third-party packages. Consider zero and negative capacity, repeated close, loader/task failures, canceled waiters, and resources released during cleanup. Avoid sleeping in tests when deterministic synchronization is possible. Explain any remaining tradeoffs after the code.'''
IGNORE_EOS=False
def one(base,model,index,round_id,limit,barrier):
 lang,task=TASKS[index%len(TASKS)]
 nonce=hashlib.sha256(f'{RUN_SALT}/{round_id}/{index}/coding-c16-v1'.encode()).hexdigest()[:20]  # Eke Hive: the run salt in the nonce — hive ignores cache_salt, so the prompt itself must differ per run
 prompt=f'Coding exercise {nonce}. Language: {lang}.\n\n{task}\n\n{GUIDANCE}\n\nUse a distinct example key or task label example_{index} in the tests.'
 body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':42,'max_tokens':limit,'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False},'cache_salt':RUN_SALT+f'-{round_id}-{index}'}
 if IGNORE_EOS:body['ignore_eos']=True
 barrier.wait();start=time.perf_counter();first=None;last=None;parts=[];usage=None;finish=None
 try:
  with request(base,'/v1/chat/completions',body) as response:
   for event in sse(response):
    if event.get('usage'):usage=event['usage']
    for choice in event.get('choices',[]):
     text=(choice.get('delta') or {}).get('content') or ''
     if text:
      now=time.perf_counter();first=now if first is None else first;last=now;parts.append(text)
     if choice.get('finish_reason'):finish=choice['finish_reason']
  end=time.perf_counter();assert usage and usage.get('completion_tokens',0)>0 and parts,'missing token usage or content'
  n=usage['completion_tokens']
  cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens')
  assert cached==0, f'Cold prefix not verified: {cached}'
  return {'status':'ok','request':index,'round':round_id,'language':lang,'prompt':prompt,'output':''.join(parts),'usage':usage,'finish_reason':finish,'elapsed_s':end-start,'ttft_ms':(first-start)*1000,'output_tokens':n,'request_output_tps':n/(end-start),'apparent_decode_tps':(n-1)/(last-first) if last and last>first else None}
 except Exception as exc:return {'status':'error','request':index,'round':round_id,'error':str(exc),'elapsed_s':time.perf_counter()-start}
def batch(base,model,c,round_id,limit):
 barrier=threading.Barrier(c+1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
  futures=[pool.submit(one,base,model,i,round_id,limit,barrier) for i in range(c)]
  start=time.perf_counter();barrier.wait();rows=[f.result() for f in futures];wall=time.perf_counter()-start
 assert all(r['status']=='ok' for r in rows),[r for r in rows if r['status']!='ok']
 return rows,wall
def main():
 global RUN_SALT
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--base-url',default='http://127.0.0.1:8430');p.add_argument('--model',default='hive');p.add_argument('--ignore-eos',action='store_true');p.add_argument('--concurrency',nargs='+',type=int,default=[1,4,8]);p.add_argument('--rounds',type=int,default=3);p.add_argument('--max-tokens',type=int,default=1536);a=p.parse_args();global IGNORE_EOS;IGNORE_EOS=a.ignore_eos
 a.output.mkdir(parents=True,exist_ok=False);RUN_SALT=hashlib.sha256(str(time.time_ns()).encode()).hexdigest()
 meta={'gpu_count':1,'concurrency':a.concurrency,'rounds':a.rounds,'max_output_tokens':a.max_tokens,'natural_eos':True,'thinking':False,'base_url':a.base_url,'model':a.model,'cache_salt':RUN_SALT,'power_limits':subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=index,power.limit','--format=csv'],text=True),'started_unix':time.time(),'aggregate_definition':'Sum of completion tokens / simultaneous-batch wall time, including prompt processing and client/transport overhead; gateway overhead applies only when the base URL is a gateway.','workload':'Python/TypeScript/Rust/Go implementation and tests; same prompts across power settings, unique per-request cache salt; cache hits must be zero.'}
 (a.output/'metadata.json').write_text(json.dumps(meta,indent=2));groups=[];all_rows=[]
 for c in a.concurrency:
  batch(a.base_url,a.model,c,-1,256);print(f'C{c} warmup complete',flush=True)
  for repeat in range(a.rounds):
   # Salt includes C to prevent cache reuse across concurrency dimensions.
   RUN_SALT=meta['cache_salt']+f'-c{c}'
   rows,wall=batch(a.base_url,a.model,c,repeat,a.max_tokens);delta={}
   for r in rows:r['concurrency']=c
   group={'round':repeat,'concurrency':c,'wall_s':wall,'output_tokens':sum(r['output_tokens'] for r in rows),'aggregate_output_tps':sum(r['output_tokens'] for r in rows)/wall,'median_ttft_ms':statistics.median(r['ttft_ms'] for r in rows),'median_request_output_tps':statistics.median(r['request_output_tps'] for r in rows),'median_apparent_decode_tps':statistics.median(r['apparent_decode_tps'] for r in rows if r['apparent_decode_tps']),'finish_reasons':{key:sum(r['finish_reason']==key for r in rows) for key in set(r['finish_reason'] for r in rows)},'backend_counter_deltas':delta}
   groups.append(group);all_rows.extend(rows);(a.output/'requests.json').write_text(json.dumps(all_rows,indent=2));(a.output/'groups.json').write_text(json.dumps(groups,indent=2));print(json.dumps(group),flush=True)
 summaries=[]
 for c in a.concurrency:
  gs=[g for g in groups if g['concurrency']==c];rs=[r for r in all_rows if r['concurrency']==c]
  summaries.append({'concurrency':c,'median_aggregate_output_tps':statistics.median(g['aggregate_output_tps'] for g in gs),'pooled_aggregate_output_tps':sum(g['output_tokens'] for g in gs)/sum(g['wall_s'] for g in gs),'round_tps':[g['aggregate_output_tps'] for g in gs],'requests':len(rs),'total_output_tokens':sum(r['output_tokens'] for r in rs),'mean_prompt_tokens':statistics.mean(r['usage']['prompt_tokens'] for r in rs),'min_prompt_tokens':min(r['usage']['prompt_tokens'] for r in rs),'max_prompt_tokens':max(r['usage']['prompt_tokens'] for r in rs),'length_capped_requests':sum(r['finish_reason']=='length' for r in rs),'median_apparent_decode_tps':statistics.median(r['apparent_decode_tps'] for r in rs if r['apparent_decode_tps'])})
 summary={'gpu_count':1,'summaries':summaries,'quality_validation':'Generation throughput only; generated coding tests were not executed.'};(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY '+json.dumps(summary),flush=True)
if __name__=='__main__':main()
