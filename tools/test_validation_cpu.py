#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""False-pass and chronology regressions. All inputs are tiny synthetic fixtures."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parent
def load(name):
    sp=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
gold=load('compare_golden');trace=load('analyze_expert_trace');mtp=load('analyze_mtp_log')

def tile_checker():
    return (ROOT/'test_tile.sh').read_text().split("<<'PY'\nimport sys, os, numpy as np",1)[1].rsplit('\nPY',1)[0].join(["import sys, os, numpy as np",""])

class Tests(unittest.TestCase):
    def test_numeric_contract(self):
        for actual,expected,ok in [(np.zeros(2),np.zeros(2),True),(np.ones(2),np.ones(2),True),
                                   (np.array([np.nan]),np.ones(1),False),(np.ones(1),np.ones(2),False),
                                   (np.zeros(2),np.ones(2),False),(None,np.ones(2),False)]:
            a=SimpleNamespace(errors=[],max_rel=.05,min_cos=.999)
            gold.check_array(a,'fixture',actual,expected)
            self.assertEqual(not a.errors,ok)

    def test_missing_and_nan_golden(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            np.savez(p/'prefill.npz',h_L00=np.ones((1,4,2),dtype=np.float32))
            (p/'prefill.json').write_text(json.dumps({'ids':[1]}))
            cmd=[sys.executable,str(ROOT/'compare_golden.py'),'--golden',td,'--dump',td]
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            np.full(8,0x7fc0,dtype=np.uint16).tofile(p/'s0_h_L0.bf16')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            np.full(8,0x3f80,dtype=np.uint16).tofile(p/'s0_h_L0.bf16')
            self.assertEqual(subprocess.run(cmd,capture_output=True).returncode,0)

    def test_empty_run_and_tile(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertNotEqual(subprocess.run([sys.executable,str(ROOT/'compare_runs.py'),td,td],capture_output=True).returncode,0)
            self.assertNotEqual(subprocess.run([sys.executable,'-',td,'33'],input=tile_checker(),text=True,capture_output=True).returncode,0)

    def test_tile_expects_decode_plus_one(self):
        # D4: hive --decode N prints N+1 tokens (hive_main.cpp: push per step + final push); the expected length comes from DECODE
        sh=(ROOT/'test_tile.sh').read_text()
        decode=int(re.search(r'^DECODE=(\d+)',sh,re.M).group(1))
        self.assertIn('--decode $DECODE',sh);self.assertIn('"$((DECODE + 1))"',sh)
        src=tile_checker()
        for n,ok in ((decode+1,True),(decode,False)):
            with tempfile.TemporaryDirectory() as td:
                for name in 'ABCDE':
                    (Path(td)/(name+'.out')).write_text('generated: '+' '.join(['7']*n)+'\n')
                    if name in 'ABC': np.arange(1,9,dtype=np.float32).tofile(Path(td)/(name+'.logits'))
                rc=subprocess.run([sys.executable,'-',td,str(decode+1)],input=src,text=True,capture_output=True).returncode
                self.assertEqual(rc==0,ok,(n,rc))

    def test_graph_fuse_compare_status_not_swallowed(self):
        # review: compare_runs.py's exit status was lost in "| grep"; now pipefail + FAIL collection + exit $FAIL
        for script in ('test_graph.sh','test_fuse.sh'):
            sh=(ROOT/script).read_text()
            self.assertRegex(sh,r'(?m)^set -[a-z]*o pipefail');self.assertRegex(sh,r'(?m)^exit \$FAIL$')
            line=[l for l in sh.splitlines() if 'compare_runs.py' in l and not l.lstrip().startswith('#')]
            self.assertEqual(len(line),1,script)
            for rc in (0,1):
                prog='set -uo pipefail\nFAIL=0\n'+line[0].strip().replace('python3 /hive/tools/compare_runs.py $A $B',f"sh -c 'echo route_ids x ok; exit {rc}'")+'\nexit $FAIL\n'
                self.assertEqual(subprocess.run(['bash','-c',prog],capture_output=True).returncode,rc,(script,rc))

    def test_spec_oracle_lands_in_golden_tag(self):
        # review: the oracle ran into a mktemp dir and ignored golden_tag; SKIP_ORACLE=1 later looked in /out/golden/$TAG
        sh=(ROOT/'test_spec.sh').read_text()
        block=sh.split('if [ "${SKIP_ORACLE:-0}" != 1 ]; then\n',1)[1].split('\nfi\n',1)[0]
        block=re.sub(r'nice -n 19 python3 /hive/oracle/dsv41_oracle.py [^\n]*',lambda _: 'touch "$TMPG/prefill.json"',block)
        with tempfile.TemporaryDirectory() as td:
            prog=('set -euo pipefail\nTAG=spec9\nG=/out/golden/$TAG\n'+block+'\n').replace('/out/',td+'/')
            for _ in range(2):
                subprocess.run(['bash','-c',prog],check=True,capture_output=True)
            self.assertTrue((Path(td)/'golden/spec9/prefill.json').exists())
            self.assertEqual(len(list((Path(td)/'golden').glob('spec9.old.*'))),1)
            self.assertFalse(list((Path(td)/'golden').glob('.spec-*')))

    def test_epoch_reuse(self):
        events={1:[(10,0,np.array([1])),(11,0,np.array([1])),(10,0,np.array([2])),(11,0,np.array([2]))]}
        self.assertEqual(trace.reuse_stats(events,1)[0][1],1)

    def test_mtp_conditional(self):
        log=''.join(f'[mtp] pos 3 draft 1 ms · verify 4 rows 2 ms · accepted {a}/3 · conf [1,2,3] · hit 5 cpu 6\n' for a in (0,3))
        log+='[mtp] pos 4 draft 2 ms · no draft (conf0 -1.0, single 3 ms)\n'
        out=io.StringIO()
        with patch.object(sys,'argv',['analyze','fake']),patch('builtins.open',return_value=io.StringIO(log)),patch('sys.stdout',out): mtp.main()
        text=out.getvalue();self.assertIn('(+1 no-draft)',text)
        # review: "single 3 ms" is hived's EMA estimate (ema_single_ms), not observed — observed = 6 tokens / (draft 4 + verify 4) ms
        self.assertIn('observed-log tokens/s 750.0',text);self.assertIn('ESTIMATE',text)
        self.assertIn('50.0%',text);self.assertIn('100.0%',text)

    def test_spec_nonfinite(self):
        spec=load('compare_spec')
        self.assertEqual(spec.stats(np.array([np.nan]),np.ones(1))[0],float('inf'))

    def test_quality_metrics(self):
        q=load('compare_quality')
        r=np.array([[1.,2.,3.],[3.,2.,1.]])
        m=q.metrics(r,r,np.array([2,0]));self.assertEqual(m['kl_max'],0);self.assertEqual(m['nll_delta'],0)
        m=q.metrics(r,-r,np.array([2,0]));self.assertGreater(m['kl_max'],0);self.assertGreater(m['nll_delta'],0)
        with self.assertRaises(ValueError): q.metrics(r,r[:1])

    def test_matrix_is_dry_by_default(self):
        p=subprocess.run([sys.executable,str(ROOT/'validation_matrix.py')],capture_output=True,text=True,check=True)
        plan=json.loads(p.stdout);self.assertFalse(plan['gpu_executed']);self.assertIn(16384,plan['lengths'])

    def test_actual_cache_replay_contract(self):
        m=load('check_cache_events')
        events=['B,2,8,0','U,3,-1,0','P,3,0,1','U,3,-1,2','C,3,0,3','U,3,0,4','E,3,0,5','I,4,0,6']
        result=m.replay(events);self.assertEqual(result['counts']['hit'],1);self.assertEqual(result['pending_observed_ms_max'],2)
        with self.assertRaises(ValueError): m.replay(events[:3]+['P,3,1,2'])
        with self.assertRaises(ValueError): m.replay(events[:3]+['U,3,0,2'])

    def test_supervisor_reaps_sibling(self):
        original=(ROOT.parent/'scripts/hive-entrypoint.sh').read_text()
        for daemon_fails in (True,False):
            fail="python3 -c 'import time,sys;time.sleep(.05);sys.exit(7)'"
            wait="python3 -c 'import time;time.sleep(60)'"
            # the daemon line runs "$BIN_DIR/$BIN" (BIN = HIVE_DAEMON_BIN: hived, or hived_glm for the GLM family)
            src=re.sub(r'^"\$BIN_DIR/\$BIN"[^\n]+? >>',lambda _: (fail if daemon_fails else wait)+' >>',original,flags=re.M)
            self.assertNotEqual(src,original,'daemon line not found in hive-entrypoint.sh')
            src=re.sub(r'^python3 /hive/server/[^\n]+? >>',lambda _: (wait if daemon_fails else fail)+' >>',src,flags=re.M)
            with tempfile.TemporaryDirectory() as td:
                src=src.replace('/out/logs/',td+'/')
                result=subprocess.run(['bash'],input=src,text=True,capture_output=True,timeout=3)
                self.assertEqual(result.returncode,7,result.stderr)
                self.assertIn('[supervisor] build dir=/hive/engine/build binary=/hive/engine/build/hived',result.stderr)

    def test_service_launcher_ignores_leftover_hive_build(self):
        # review: an exported HIVE_BUILD=build-dev left in the operator shell silently ran an unpromoted binary as the service.
        opts=load('test_options_cpu')  # runs scripts/hive-start.sh for real against stub docker
        with tempfile.TemporaryDirectory() as td:
            got=opts.forwarded(opts.run_launcher(td,'hive-start.sh',(),{'HIVE_BUILD':'build-dev','HIVE_X':'1'})[0])
            self.assertEqual(got.get('HIVE_BUILD'),'build');self.assertEqual(got.get('HIVE_X'),'1')
            got=opts.forwarded(opts.run_launcher(td,'hive-start.sh',('--build','build-dev'),{})[0])
            self.assertEqual(got.get('HIVE_BUILD'),'build-dev')
        env={'PATH':'/usr/bin:/bin','HIVE_BUILD':'build-dev','HIVE_X':'1'}
        ep=(ROOT.parent/'scripts/hive-entrypoint.sh').read_text()
        with tempfile.TemporaryDirectory() as td:
            src='\n'.join(l for l in ep.splitlines() if l.startswith(('BIN_','msg=','[ "${HIVE_BUILD','echo "$msg"'))).replace('/out/logs/',td+'/')
            r=subprocess.run(['bash','-c',src],env={**env},capture_output=True,text=True)
            self.assertIn('build dir=/hive/engine/build-dev',r.stderr);self.assertIn('(not the default engine/build)',r.stderr)

if __name__=='__main__': unittest.main()
