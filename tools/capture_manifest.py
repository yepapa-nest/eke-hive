#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Read-only provenance; weights are stat-only unless --hash-weights is explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def command(argv):
    try: return subprocess.run(argv,text=True,capture_output=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired): return None

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--binary',required=True);cli.add_argument('--weights',required=True)
    cli.add_argument('--hash-weights',action='store_true');cli.add_argument('--input',action='append',default=[])
    cli.add_argument('--cache-state');cli.add_argument('--seed',type=int,default=0)
    cli.add_argument('--cache-mode',choices=['cold','warm','unknown'],default='unknown')
    a=cli.parse_args();root=Path(__file__).resolve().parents[1];weights=Path(a.weights)
    paths=sorted(p for base in ('engine','server','oracle','tools','ws') for p in (root/base).rglob('*')
                 if p.is_file() and not any(x.startswith('build') or x=='__pycache__' or x=='third_party' for x in p.parts)
                 and p.suffix in ('.h','.cpp','.cu','.py','.sh','.txt'))
    source=hashlib.sha256(''.join(f'{p.relative_to(root)}:{digest(p)}\n' for p in paths).encode()).hexdigest()
    files={}
    for f in sorted(weights.glob('*')):
        if not f.is_file() or f.suffix not in ('.json','.safetensors','.py'): continue
        st=f.stat();entry={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
        if f.suffix!='.safetensors' or a.hash_weights: entry['sha256']=digest(f)
        files[f.name]=entry
    result={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_sha256':source,
            'binary':str(Path(a.binary).resolve()),'binary_sha256':digest(a.binary),'python':sys.version,'platform':platform.platform(),
            'weights':files,'weight_content_verified':a.hash_weights,'seed':a.seed,'cache_mode':a.cache_mode,
            'cache_state_sha256':digest(a.cache_state) if a.cache_state else None,
            'inputs':{f:digest(f) for f in a.input},'environment_overrides':{k:v for k,v in os.environ.items() if k.startswith(('HIVE_','OMP_','OPENBLAS_')) and not any(s in k for s in ('SECRET','TOKEN','KEY','PASS'))},
            'gpu':command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.used,memory.total,power.limit','--format=csv']),
            'numa':command(['numactl','--hardware']),
            'note':'Capture effective daemon argv/stats alongside this manifest; environment overrides are NOT resolved runtime defaults.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
