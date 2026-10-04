#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Replay actual HIVE_CACHE_EVENTS transitions; NOT a policy prediction.
Buffered/truncated traces are not a complete run. Use a short clean capture first.
"""
import argparse
import collections
import json

def replay(lines):
    resident={};mapping={};pending={};pending_keys={};begun=False;slots=0
    counts=collections.Counter();pending_ms=[]
    for line_no,line in enumerate(lines,1):
        fields=line.rstrip('\n').split(',')
        if len(fields)!=4: raise ValueError(f'line {line_no}: incomplete event')
        op=fields[0];key,slot=int(fields[1]),int(fields[2]);ms=float(fields[3])
        counts[op]+=1
        if op=='B':
            resident.clear();mapping.clear();pending.clear();pending_keys.clear();slots=key;begun=True;continue
        if not begun: raise ValueError('missing initial B event')
        def require(condition,message):
            if not condition: raise ValueError(f'line {line_no}: {message}')
        require(slot==-1 if op=='U' and slot<0 else 0<=slot<slots,'invalid slot')
        if op=='U':
            require(mapping.get(key,-1)==slot,'observed hit differs from replay')
            counts['hit' if slot>=0 else 'miss']+=1
        elif op=='E':
            require(resident.get(slot)==key and mapping.get(key)==slot,'eviction mapping mismatch')
            del resident[slot];del mapping[key]
        elif op=='P':
            require(key not in mapping and key not in pending_keys,'duplicate resident/pending key')
            require(slot not in resident and slot not in pending,'occupied promotion slot')
            pending[slot]=(key,ms);pending_keys[key]=slot
        elif op in ('C','D'):
            require(slot in pending and pending[slot][0]==key,'commit without matching issue')
            pending_ms.append(ms-pending.pop(slot)[1]);del pending_keys[key]
            if op=='C':
                require(key not in mapping,'duplicate commit');resident[slot]=key;mapping[key]=slot
        elif op=='I':
            require(key not in mapping and key not in pending_keys and slot not in resident and slot not in pending,'duplicate place')
            resident[slot]=key;mapping[key]=slot
        else: raise ValueError(f'unknown event {op}')
    if not begun: raise ValueError('empty trace')
    return {'ok':True,'counts':dict(counts),'resident':len(resident),'pending':len(pending),
            'observed_hit_rate':counts['hit']/max(1,counts['hit']+counts['miss']),
            'pending_observed_ms_max':max(pending_ms,default=0),
            'note':'All observed phases combined; excludes buffered/lost tail; observed commit latency is not pure DMA duration.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace');a=p.parse_args()
    try:
        with open(a.trace) as f: result=replay(f)
    except (ValueError,OSError) as e: print(json.dumps({'ok':False,'error':str(e)}));return 1
    print(json.dumps(result));return 0
if __name__=='__main__': raise SystemExit(main())
