#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""CPU test of server/speech/tts_server.py with a fake engine (no GPU, no model): the /ws/speak protocol (PCM chunks then done, empty text,
the text-proportional length cap, per-request sampling (both stages, then the code predictor alone), silence (pauses kept up to
--silence-keep-ms, a --silence-stop-ms run ends the take),
error then done, unknown voice / language), barge-in (closing the socket stops the generation), queued connections, /health, 404, a
fragmented and a masked-large client frame, and the voice / language resolution rules of the reference (auto = no language id, dialect
speakers under chinese / auto, dialect names rejected as languages). Uses its own minimal WebSocket client (stdlib only)."""
import asyncio
import base64
import json
import os
import struct
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server", "speech"))
import tts_server as T  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILS.append(name)


# ---- minimal client ------------------------------------------------------------------------------------------------------------------------
class Client:
    @classmethod
    async def connect(cls, port, path="/ws/speak"):
        c = cls()
        c.r, c.w = await asyncio.open_connection("127.0.0.1", port)
        key = base64.b64encode(os.urandom(16)).decode()
        c.w.write(f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  "Sec-WebSocket-Version: 13\r\n\r\n".encode())
        head = (await c.r.readuntil(b"\r\n\r\n")).decode()
        assert head.startswith("HTTP/1.1 101"), head
        return c

    def _frame(self, op, data, fin=True):
        mask = os.urandom(4)
        n = len(data)
        hdr = bytes([(0x80 if fin else 0) | op])
        hdr += bytes([0x80 | n]) if n < 126 else (bytes([0x80 | 126]) + struct.pack("!H", n) if n < 65536 else bytes([0x80 | 127]) + struct.pack("!Q", n))
        return hdr + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(data))

    async def send_text(self, s, fragments=1):
        b = s.encode()
        k = max(1, len(b) // fragments)
        parts = [b[i:i + k] for i in range(0, len(b), k)]
        for i, p in enumerate(parts):
            self.w.write(self._frame(0x1 if i == 0 else 0x0, p, fin=i == len(parts) - 1))
        await self.w.drain()

    async def recv(self):
        h = await self.r.readexactly(2)
        op, n = h[0] & 0x0F, h[1] & 0x7F
        if n == 126:
            n = struct.unpack("!H", await self.r.readexactly(2))[0]
        elif n == 127:
            n = struct.unpack("!Q", await self.r.readexactly(8))[0]
        data = await self.r.readexactly(n)
        return (op, data.decode() if op == 0x1 else data)

    async def messages(self):
        """binary payloads until the done message"""
        pcm, msgs = [], []
        while True:
            op, d = await self.recv()
            if op == 0x2:
                pcm.append(d)
            elif op == 0x1:
                m = json.loads(d)
                msgs.append(m)
                if m["type"] == "done":
                    return pcm, msgs
            elif op == 0x8:
                return pcm, msgs

    def close(self):
        self.w.close()


# ---- fakes ---------------------------------------------------------------------------------------------------------------------------------
class FakeEngine:
    info = {"weight_mb": 1, "buffer_mb": 1, "context_mb": 1, "process_mb": 3, "max_seq": 2048, "limit_mb": 0}

    VOICED = b"\xe8\x03"  # 1000 — above --silence-rms
    SILENT = b"\x01\x00"

    def __init__(self, chunks=6, pattern=None):
        self.n, self.chunks, self.cancelled, self.run, self.starts = 0, chunks, False, 0, []
        self.pattern = pattern  # "v"/"s" per chunk; None = all voiced
        self.temps = []

    def start(self, ids, ins, sid, lid, seed, sampling):
        self.starts.append((sid, lid, ins, sampling.max_frames, len(ids)))
        self.temps.append((round(sampling.temperature, 3), round(sampling.sub_temperature, 3), sampling.top_k, sampling.sub_top_k))
        self.n = len(self.pattern) if self.pattern else self.chunks

    def next_chunk(self, cancel):
        for _ in range(4):  # 4 frames per chunk, cancel checked before each
            if cancel.value:
                self.cancelled = True
                return None
            time.sleep(0.01)
        if self.n == 0:
            return None
        self.n -= 1
        self.run += 1
        if self.pattern:
            kind = self.pattern[len(self.pattern) - self.n - 1]
            return (self.VOICED if kind == "v" else self.SILENT) * T.CHUNK_SAMPLES
        return self.VOICED * (T.CHUNK_SAMPLES if self.n else 1920)


def make_prompter():
    d = tempfile.mkdtemp()
    cfg = {"talker_config": {"spk_id": {"serena": 3066, "sohee": 2864, "eric": 2875, "dylan": 2878},
                             "spk_is_dialect": {"serena": False, "sohee": False, "eric": "sichuan_dialect", "dylan": "beijing_dialect"},
                             "codec_language_id": {"chinese": 2055, "english": 2050, "korean": 2064, "beijing_dialect": 2074,
                                                   "sichuan_dialect": 2062}}}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    orig = T.Prompter._tokenizer
    T.Prompter._tokenizer = staticmethod(lambda model_dir: (lambda s: [len(s)] * 10))
    try:
        return T.Prompter(d)
    finally:
        T.Prompter._tokenizer = orig


def test_resolve(p):
    check("resolve korean", p.resolve("Sohee", "Korean") == (2864, 2064))
    check("resolve auto: no language id", p.resolve("sohee", "Auto") == (2864, -1))
    check("resolve dialect speaker under auto", p.resolve("dylan", "auto") == (2878, 2074))
    check("resolve dialect speaker under chinese", p.resolve("Eric", "Chinese") == (2875, 2062))
    check("resolve dialect speaker under korean", p.resolve("eric", "Korean") == (2875, 2064))
    for v, l in (("nobody", "Korean"), ("sohee", "klingon"), ("sohee", "beijing_dialect")):
        try:
            p.resolve(v, l)
            check(f"reject {v}/{l}", False)
        except ValueError:
            check(f"reject {v}/{l}", True)
    check("speakers sorted", p.speakers == sorted(p.speakers))


async def run():
    p = make_prompter()
    test_resolve(p)
    a = types.SimpleNamespace(model="m", lib=None, vram_limit_mb=0, max_seq=0, seed=-1, temperature=0.9, top_k=50, rep_penalty=1.05,
                              max_frames_base=20, max_frames_per_token=8, silence_rms=328.0, silence_keep_ms=960, silence_stop_ms=3200)
    srv = T.Server(a)
    srv.engine, srv.prompter, srv.ready = FakeEngine(), p, True
    server = await asyncio.start_server(srv.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "안녕하세요", "voice": "Sohee", "language": "Auto"}), fragments=3)
    pcm, msgs = await c.messages()
    total = 5 * T.CHUNK_SAMPLES + 1920
    check("stream: chunk sizes", [len(x) for x in pcm] == [2 * T.CHUNK_SAMPLES] * 5 + [2 * 1920])
    d = msgs[-1]
    check("stream: done", d["type"] == "done" and d["audio_ms"] == round(total / 24) and d["total_ms"] >= d["first_ms"] > 0, str(d))
    check("stream: auto language passed as -1", srv.engine.starts[-1][:2] == (2864, -1))
    st = srv.engine.starts[-1]
    check("length cap = 20 + 8 per text token", st[3] == 20 + 8 * (st[4] - 8) and st[3] > 0, str(st[3:]))
    srv.a.max_frames_per_token = 0
    c2 = await Client.connect(port)
    await c2.send_text(json.dumps({"type": "speak", "text": "x"}))
    await c2.messages()
    c2.close()
    check("length cap off (0 per token)", srv.engine.starts[-1][3] == 0)
    srv.a.max_frames_per_token = 8
    c.close()

    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "  "}))
    _, msgs = await c.messages()
    check("empty text: done only", msgs == [{"type": "done", "audio_ms": 0}], str(msgs))
    c.close()

    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "x", "voice": "nobody"}))
    _, msgs = await c.messages()
    check("bad voice: error then done", [m["type"] for m in msgs] == ["error", "done"] and "nobody" in msgs[0]["message"], str(msgs))
    c.close()

    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "x", "instruct": "y" * 70000}))  # a > 64 KiB masked frame
    pcm, msgs = await c.messages()
    check("large request frame + instruct", msgs[-1]["type"] == "done" and srv.engine.starts[-1][2] is not None)
    c.close()

    # per-request sampling: temperature sets both stages, sub_temperature / sub_top_k then the code predictor alone
    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "x", "temperature": 0.4}))
    await c.messages()
    c.close()
    check("temperature sets both stages", srv.engine.temps[-1] == (0.4, 0.4, 50, 50), str(srv.engine.temps[-1]))
    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "x", "temperature": 0.4, "sub_temperature": 0.9, "sub_top_k": 30}))
    await c.messages()
    c.close()
    check("sub_temperature / sub_top_k set the code predictor alone", srv.engine.temps[-1] == (0.4, 0.9, 50, 30), str(srv.engine.temps[-1]))
    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "x", "sub_temperature": 9, "sub_top_k": True}))
    await c.messages()
    c.close()
    check("out-of-range sub_* keep the server's", srv.engine.temps[-1] == (0.9, 0.9, 50, 50), str(srv.engine.temps[-1]))

    # silence: a natural pause passes, a long one is shortened to silence_keep_ms (3 chunks) and 10 silent chunks (3.2 s) end the take
    C = 2 * T.CHUNK_SAMPLES
    for name, pattern, want_chunks, want_stop, want_dropped in (
            ("pause of 2 chunks sent whole", "vssv", 4, False, 0),
            ("pause of 6 chunks shortened to 3", "vssssssv", 5, False, 960),
            ("10 silent chunks end the take", "v" + "s" * 14 + "v", 4, True, 7 * 320),
            ("no speech at all ends the take, nothing sent", "s" * 14, 0, True, 7 * 320)):
        srv.engine = FakeEngine(pattern=pattern)
        c = await Client.connect(port)
        await c.send_text(json.dumps({"type": "speak", "text": "x"}))
        pcm, msgs = await c.messages()
        c.close()
        d = msgs[-1]
        ok = (len(pcm) == want_chunks and all(len(x) == C for x in pcm) and d["type"] == "done"
              and (d.get("stopped") == "silence") == want_stop and d.get("dropped_ms", 0) == want_dropped
              and d["audio_ms"] == want_chunks * 320 and (not want_stop or srv.engine.run <= 11))
        check(f"silence: {name}", ok, f"chunks {len(pcm)} done {d} run {srv.engine.run}")

    srv.engine = FakeEngine(chunks=50)
    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "long"}))
    await c.recv()
    c.close()
    await asyncio.sleep(0.3)
    check("barge-in: generation stopped", srv.engine.cancelled and srv.engine.run <= 3, f"chunks run {srv.engine.run}")

    srv.engine = FakeEngine(chunks=3)

    async def one():
        cl = await Client.connect(port)
        await cl.send_text(json.dumps({"type": "speak", "text": "x"}))
        r = await cl.messages()
        cl.close()
        return r[1][-1]

    r = await asyncio.gather(one(), one())
    check("two clients queue", all(m["type"] == "done" and m["audio_ms"] == round((2 * T.CHUNK_SAMPLES + 1920) / 24) for m in r) and
          srv.engine.run == 6, str(r))

    rd, wr = await asyncio.open_connection("127.0.0.1", port)
    wr.write(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
    await wr.drain()
    body = json.loads((await rd.read()).decode().split("\r\n\r\n", 1)[1])
    check("health", body["ok"] is True and body["speakers"] == p.speakers, str(body)[:120])
    rd, wr = await asyncio.open_connection("127.0.0.1", port)
    wr.write(b"GET /nope HTTP/1.1\r\nHost: x\r\n\r\n")
    await wr.drain()
    check("404", (await rd.read()).startswith(b"HTTP/1.1 404"))

    srv.ready = False
    c = await Client.connect(port)
    await c.send_text(json.dumps({"type": "speak", "text": "x"}))
    _, msgs = await c.messages()
    check("not loaded: error then done", [m["type"] for m in msgs] == ["error", "done"])
    server.close()


asyncio.run(run())
print("all passed" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
