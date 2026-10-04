#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Converts the [decode-host] · [cache] · [mtp] · [profile] lines of hived.log into per-step JSONL (verify steps take hit/cpu from
the following [mtp] pos line).
The JSONL is used to regress CPU expert time on the number of (expert, row) items (measured: 0.18–0.20 ms/item), and to measure
the per-layer front segment and GPU idle time by batch and verify.
Usage: python3 tools/decode_host_steps.py hived.log [archive/...] out.jsonl   (on the engine host — nice -n 19 recommended)
"""
import re,sys,json
rdh=re.compile(r"\[decode-host M=(\d+)( verify)?\] sync ([\d.]+) · prep ([\d.]+) · launch ([\d.]+) · cpu ([\d.]+) vs gpu ([\d.]+) \(cpu-bound layers (\d+)/40\) · tail ([\d.]+) ms · next ([\d.]+) · front ([\d.]+) · post ([\d.]+) · hprep ([\d.]+) · gpu-idle ([\d.]+) of ([\d.]+) ms \(cpu layers (\d+)\)")
rc=re.compile(r"\[cache\] step \d+ M=(\d+) routed (\d+) hit (\d+) cpu (\d+) streamed (\d+)")
rm=re.compile(r"\[mtp\] pos \d+ draft [\d.]+ ms · verify (\d+) rows [\d.]+ ms .*hit (\d+) cpu (\d+)")
rp=re.compile(r"\[profile M=(\d+)\] total ([\d.]+) ms: (.*?) · dma\.span ([\d.]+) dma\.wait (\S+) ms \((\d+) layers\)")
out=open(sys.argv[-1],"w")
for fn in sys.argv[1:-1]:
  last_cache=None; pend=None; prof=None
  for l in open(fn,errors="replace"):
    if l.startswith("[cache]"):
      m=rc.match(l)
      if m: last_cache=[int(x) for x in m.groups()]
      continue
    if l.startswith("[profile M="):
      m=rp.match(l)
      if m:
        d=dict((k,float(v)) for k,v in re.findall(r"(\S+) ([\d.]+)",m[3])); prof=dict(M=int(m[1]),total=float(m[2]),dma_span=float(m[4]),dma_layers=int(m[6]),**d)
      continue
    if l.startswith("[decode-host"):
      m=rdh.match(l)
      if not m: continue
      g=m.groups()
      rec=dict(M=int(g[0]),verify=bool(g[1]),sync=float(g[2]),prep=float(g[3]),launch=float(g[4]),cpu=float(g[5]),gpu=float(g[6]),cpubound=int(g[7]),tail=float(g[8]),front=float(g[10]),idle=float(g[13]),step=float(g[14]),cpulayers=int(g[15]),prof=prof)
      prof=None
      if rec["verify"]: pend=rec
      else:
        if last_cache and last_cache[0]==rec["M"]: rec.update(routed=last_cache[1],hit=last_cache[2],ncpu=last_cache[3],streamed=last_cache[4])
        out.write(json.dumps(rec)+"\n")
      continue
    if pend is not None and l.startswith("[mtp] pos"):
      m=rm.match(l)
      if m and int(m[1])==pend["M"]:
        pend.update(hit=int(m[2]),ncpu=int(m[3]),routed=pend["M"]*240)
      out.write(json.dumps(pend)+"\n"); pend=None
