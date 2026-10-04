#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Download only the FP8 tensors of the original Z.ai GLM-5.3-Flash release that hived_glm uses next to the NVFP4 checkpoint:
the DSA attention projections (q_a, q_b, kv_a_with_mqa, o of the 11 DSA layers + the MTP layer) and the shared experts, each with its
128×128 block scale (weight_scale_inv). About 2.1 GiB, fetched with HTTP range requests from the release's safetensors shards into one
local safetensors file plus a model.safetensors.index.json — point HIVE_GLM_FP8_DIR at the output directory.
--mtp-experts additionally fetches the routed experts of the MTP layer (layer 45, 288 × gate/up/down FP8 + scales, about 7 GiB) into a
second file (mtp_experts.safetensors) of the same directory: the NVFP4 checkpoint ships them in BF16 only.

  python3 scripts/glm-fetch-fp8.py /models/GLM-5.3-Flash-zai-fp8 [--mtp-experts] [--repo zai-org/GLM-5.3-Flash] [--revision main]
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request

P = argparse.ArgumentParser(description="Fetch the Z.ai FP8 tensors that hived_glm uses next to the NVFP4 checkpoint.")
P.add_argument("out_dir", help="output directory (point HIVE_GLM_FP8_DIR at it)")
P.add_argument("--repo", default="zai-org/GLM-5.3-Flash", help="Hugging Face repository of the original FP8 release")
P.add_argument("--revision", default="main", help="revision of that repository")
P.add_argument("--mtp-experts", action="store_true", help="also fetch the MTP layer's routed experts in FP8 (about 7 GiB)")
A = P.parse_args()
BASE = f"https://huggingface.co/{A.repo}/resolve/{A.revision}/"


def get(url, rng=None, tries=5):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % rng} if rng else {})
            return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:  # noqa: BLE001 — transient network errors are retried
            print("retry", url, e, flush=True)
            time.sleep(3 * (t + 1))
    sys.exit("failed: " + url)


def wanted(name: str, dtype: str) -> bool:
    if ".layers." not in name or dtype == "BF16":
        return False
    if "shared_experts" in name:
        return True
    return "self_attn" in name and any(p in name for p in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", ".o_proj"))


index = json.loads(get(BASE + "model.safetensors.index.json"))
shards = sorted(set(index["weight_map"].values()))
headers = {}
for f in shards:  # each shard: 8-byte header length + JSON header
    n = struct.unpack("<Q", get(BASE + f, (0, 7)))[0]
    h = json.loads(get(BASE + f, (8, 7 + n)))
    for k, v in h.items():
        if k != "__metadata__":
            headers[k] = dict(v, file=f, base=8 + n)


def fetch(names, fname):
    meta, off = {}, 0
    for k in names:
        n = headers[k]["data_offsets"][1] - headers[k]["data_offsets"][0]
        meta[k] = {"dtype": headers[k]["dtype"], "shape": headers[k]["shape"], "data_offsets": [off, off + n]}
        off += n
    hdr = json.dumps(meta).encode()
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    tmp = os.path.join(A.out_dir, fname + ".part")
    t0 = time.time()
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hdr)) + hdr)
        for i, k in enumerate(names):
            v = headers[k]
            a = v["base"] + v["data_offsets"][0]
            b = v["base"] + v["data_offsets"][1] - 1
            data = get(BASE + v["file"], (a, b))
            if len(data) != b - a + 1:
                sys.exit("short read: " + k)
            out.write(data)
            if i % 50 == 0:
                print(f"{fname}: {i}/{len(names)} {k} {time.time() - t0:.0f} s", flush=True)
    os.replace(tmp, os.path.join(A.out_dir, fname))
    print(f"{fname}: {len(names)} tensors, {off / 2**30:.2f} GiB in {time.time() - t0:.0f} s")


os.makedirs(A.out_dir, exist_ok=True)
files = {}
main_names = sorted(k for k, v in headers.items() if wanted(k, v["dtype"]))
if not os.path.exists(os.path.join(A.out_dir, "fp8.safetensors")):
    fetch(main_names, "fp8.safetensors")
files.update({k: "fp8.safetensors" for k in main_names})
if A.mtp_experts:
    n_layers = None
    for k in headers:
        if ".mlp.experts." in k:
            n_layers = max(n_layers or 0, int(k.split(".layers.")[1].split(".")[0]))
    mtp_names = sorted(k for k, v in headers.items() if f".layers.{n_layers}.mlp.experts." in k and v["dtype"] != "BF16")
    if not os.path.exists(os.path.join(A.out_dir, "mtp_experts.safetensors")):
        fetch(mtp_names, "mtp_experts.safetensors")
    files.update({k: "mtp_experts.safetensors" for k in mtp_names})
with open(os.path.join(A.out_dir, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"source": f"{A.repo}@{A.revision} (FP8 tensors used by hive_glm)"}, "weight_map": files}, f)
print("index:", len(files), "tensors ->", A.out_dir)
