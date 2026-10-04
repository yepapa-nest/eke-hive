#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Teacher-forced logits quality metrics; same token rows/vocabulary required.
NPY inputs [rows,vocab], optional target token IDs NPY [rows]. Does not run a model.
Thresholds are explicit study choices, not claims of lossless model quality.
"""
import argparse
import json
import numpy as np

def metrics(reference, candidate, targets=None):
    r=np.asarray(reference,dtype=np.float64);c=np.asarray(candidate,dtype=np.float64)
    if r.shape!=c.shape or r.ndim!=2 or not r.size or not np.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError('equal nonempty finite [rows,vocab] logits required')
    def logsoftmax(x):
        y=x-x.max(axis=-1,keepdims=True)
        return y-np.log(np.exp(y).sum(axis=-1,keepdims=True))
    lr,lc=logsoftmax(r),logsoftmax(c)
    kl=(np.exp(lr)*(lr-lc)).sum(-1)
    result={'rows':len(r),'argmax_agreement':float(np.mean(r.argmax(-1)==c.argmax(-1))),
            'kl_mean':float(kl.mean()),'kl_max':float(kl.max()),'logit_max_abs':float(np.abs(r-c).max())}
    if targets is not None:
        t=np.asarray(targets)
        if t.shape!=(len(r),) or not np.issubdtype(t.dtype,np.integer) or np.any(t<0) or np.any(t>=r.shape[1]): raise ValueError('invalid target rows')
        rows=np.arange(len(r));rn=-lr[rows,t];cn=-lc[rows,t]
        result.update(reference_nll=float(rn.mean()),candidate_nll=float(cn.mean()),nll_delta=float((cn-rn).mean()))
    return result

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference');cli.add_argument('candidate');cli.add_argument('--targets')
    cli.add_argument('--max-kl',type=float,required=True);cli.add_argument('--min-argmax',type=float,required=True)
    cli.add_argument('--max-nll-delta',type=float)
    a=cli.parse_args()
    result=metrics(np.load(a.reference),np.load(a.candidate),np.load(a.targets) if a.targets else None)
    ok=result['kl_max']<=a.max_kl and result['argmax_agreement']>=a.min_argmax
    if a.max_nll_delta is not None: ok &= 'nll_delta' in result and result['nll_delta']<=a.max_nll_delta
    print(json.dumps({**result,'ok':bool(ok)}));return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
