#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Boundary/verify regression plan. Default only prints JSON; --execute uses GPU.
Run only after the user has stopped other GPU workloads and authorized testing.
Logits parity is NOT an independent reference quality score.
"""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from compare_quality import metrics

LENGTHS=(15,16,127,128,129,1023,1024,2687,2688,2689,16383,16384,16385)
FEATURES={
    'baseline':{}, 'mtp-cache':{'HIVE_MTP_CACHE':'1'}, 'phase-score':{'HIVE_PHASE_SCORE':'1'},
    'staging-reuse':{'HIVE_CACHE_REUSE_STAGE':'1'}, 'score-victims':{'HIVE_PROMOTE_SCORE':'1'},
    'tile-group':{'HIVE_TILE_GROUP_GEMM':'1'}, 'checkpoint-delta':{'HIVE_CKPT_DELTA':'1'},
    'checkpoint-async':{'HIVE_CKPT_ASYNC':'1'}, 'idx-f32':{'HIVE_IDX_F32':'1'}, 'idx-tc':{'HIVE_IDX_TC':'1'},
}
def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',default='/hive/engine/'+os.environ.get('HIVE_BUILD','build')+'/hive');ap.add_argument('--ckpt',default=os.environ.get('HIVE_CKPT'),help='checkpoint directory (default: $HIVE_CKPT; required with --execute)')
    ap.add_argument('--engram',default='/out/engram');ap.add_argument('--out',default='/out/validation')
    ap.add_argument('--lengths',default=','.join(map(str,LENGTHS)));ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--cache-mb',type=int,default=68000);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--max-kl',type=float,default=.001);ap.add_argument('--min-argmax',type=float,default=1)
    a=ap.parse_args();lengths=[int(x) for x in a.lengths.split(',')]
    plan={'gpu_executed':False,'lengths':lengths,'variants':['whole','chunk1024','tile3'],
          'verify_prefixes':[127,128,129],'features_for_separate_AB':FEATURES,
          'service_only_tests':['cold/warm','c1/c4/c8/c16','EOS/cancel','session edit/history/changed images','P95/P99 ITL','sampling distribution'],
          'thresholds':{'max_kl':a.max_kl,'min_argmax':a.min_argmax},
          'notes':['Synthetic tokens test numerical boundaries, not language quality.','Feature matrix is a plan; this runner executes baseline chunk/tile and verify cases only.']}
    if not a.execute: print(json.dumps(plan,indent=2));return 0
    if not a.ckpt: ap.error('--ckpt DIR (or HIVE_CKPT) is required')
    Path(a.out).mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='run-',dir=a.out))
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    env={k:v for k,v in os.environ.items() if not k.startswith('HIVE_')}
    # Only explicit test environment. Avoid inherited approximation/fusion flags.
    ok=True;results=[]
    for n in lengths:
        ids=out/f'ids-{n}.txt';ids.write_text(','.join(str(1000+(i*7919+a.seed)%100000) for i in range(n)))
        common=[a.binary,'--ckpt',a.ckpt,'--engram',a.engram,'--ids-file',str(ids),
                '--max-ctx',str(n+64),'--vram-cache-mb',str(a.cache_mb),'--vision-max-patches','64']
        reference=None;reference_tokens=None
        for name,chunk,tile in [('whole',n,1),('chunk1024',1024,1),('tile3',8192,3)]:
            logits=out/f'{n}-{name}.f32'
            cmd=common+['--max-chunk',str(chunk),'--prefill-tile',str(tile),'--logits-out',str(logits),'--decode','32']
            with (out/f'{n}-{name}.log').open('w') as log: done=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            row={'length':n,'variant':name,'exit':done.returncode,'command':cmd}
            if done.returncode==0 and logits.exists():
                values=np.fromfile(logits,dtype=np.float32)[None,:]
                if reference is None: reference=values
                try:
                    row.update(metrics(reference,values))
                    row['ok']=row['kl_max']<=a.max_kl and row['argmax_agreement']>=a.min_argmax
                    matches=re.findall(r'^generated:(.*)$',(out/f'{n}-{name}.log').read_text(),re.M)
                    tokens=[int(t) for t in matches[-1].split()] if matches else []
                    if reference_tokens is None: reference_tokens=tokens
                    row['token_parity']=len(tokens)==33 and tokens==reference_tokens
                    row['ok'] &= row['token_parity']
                except ValueError as e: row.update(ok=False,error=str(e))
            else: row['ok']=False
            ok &= row['ok'];results.append(row)
            (out/'results.json').write_text(json.dumps(results,indent=2))
        if n in (127,128,129):
            cmd=common+['--max-chunk','1024','--verify-test']
            with (out/f'{n}-verify.log').open('w') as log: done=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            results.append({'length':n,'variant':'verify','exit':done.returncode,'ok':done.returncode==0,'command':cmd});ok &= done.returncode==0
    (out/'results.json').write_text(json.dumps(results,indent=2));print(out);return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
