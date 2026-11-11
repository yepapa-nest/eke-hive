// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Per-request tool-name constraint (server field "allowed_tools", docs/configuration.md): while the model writes a tool name inside a DSML
//   tool-call block, only tokens that keep the text a prefix of an allowed name (and then close the name) can be sampled.
//   Only the name span is constrained: plain text, the thinking span, whether a tool is called at all, and the arguments are the model's own.
//   Sampling only — nothing here touches the prompt, the KV cache or the session key, so a request with the field reuses exactly the prefix a
//   request without it would.
//   Tracking follows the emitted tokens and the reference grammar <｜DSML｜ calls>\n<｜DSML｜ invoke name="NAME">\n<｜DSML｜ parameter name="P"
//   string="true">V</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>. The tag pieces are single tokens the server looks up; a block
//   written with other token boundaries is simply not recognized (served unconstrained; the server's record flags a call outside the list).
//   Text inside a parameter value is not a tag: a value quoting an invoke tag is left alone.
//   Speculative verification samples each row through sample() when the row is in a name span and accepts a draft only when it equals that
//   sample, so the output distribution is the constrained one with or without MTP; the drafts themselves are not constrained.
// Host side only — no CUDA (tests/test_tool_mask_cpu.cpp).
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hive/sampler.h"

namespace hive::toolmask {

// The allowed names as a token automaton: node n has the edges tok[first[n] .. first[n + 1]); next[e] = the node after edge e, −1 = the name closed.
//   Node 0 is the start of a name. The server builds it from the tokenizer's vocabulary (server/hive_server.py compile_tool_trie).
struct Trie {
  std::vector<int32_t> first, tok, next;
  bool valid(int32_t vocab) const {
    if (first.size() < 2 || first.front() != 0 || (size_t)first.back() != tok.size() || tok.size() != next.size()) return false;
    const int32_t nodes = (int32_t)first.size() - 1;
    for (size_t i = 1; i < first.size(); ++i) if (first[i] < first[i - 1]) return false;
    for (size_t e = 0; e < tok.size(); ++e)
      if (tok[e] < 0 || tok[e] >= vocab || next[e] < -1 || next[e] >= nodes) return false;
    return true;
  }
};

// The single tokens of the grammar's tag pieces.
struct Tags { int32_t dsml = -1, calls = -1, invoke = -1, parameter = -1, name = -1, eq_quote = -1; };

enum class Mode : uint8_t { kText, kBlock, kParam, kName };

struct State {
  bool on = false;
  Trie trie;
  Tags tag;
  int32_t think_start = -1, think_end = -1;
  bool in_think = false;
  Mode mode = Mode::kText;
  uint8_t seen = 0;      // tag-piece progress: 0 none · 1 ｜DSML｜ · 2 + invoke · 3 + invoke name · 4 + parameter · 5 + parameter name
  int32_t node = 0;      // trie node while mode == kName
  long rows = 0;         // rows sampled through the constraint
  long spans = 0;        // name spans entered
  long left = 0;         // tokens in a name span outside the trie (a span whose node has no edge, a token the constraint did not sample)

  void feed(int32_t t) {
    if (!on) return;
    if (in_think) { if (t == think_end) in_think = false; return; }
    switch (mode) {
      case Mode::kName: {
        const int32_t e = edge(t);
        if (e < 0) { ++left; mode = Mode::kBlock; seen = 0; return; }
        node = trie.next[e];
        if (node < 0) { mode = Mode::kBlock; seen = 0; }
        return;
      }
      case Mode::kText:
        if (think_start >= 0 && t == think_start) { in_think = true; seen = 0; return; }
        if (seen == 1 && t == tag.calls) { mode = Mode::kBlock; seen = 0; return; }
        seen = t == tag.dsml ? 1 : 0;
        return;
      case Mode::kParam:   // a value ends at ｜DSML｜ parameter without a name attribute (the closing tag)
        if (seen == 1) seen = t == tag.parameter ? 2 : (t == tag.dsml ? 1 : 0);
        else if (seen == 2) { if (t != tag.name) mode = Mode::kBlock; seen = t == tag.dsml ? 1 : 0; }
        else seen = t == tag.dsml ? 1 : 0;
        return;
      case Mode::kBlock:
        switch (seen) {
          case 1:
            if (t == tag.calls) { mode = Mode::kText; seen = 0; }
            else if (t == tag.invoke) seen = 2;
            else if (t == tag.parameter) seen = 4;
            else seen = t == tag.dsml ? 1 : 0;
            return;
          case 2: case 4:
            if (t == tag.name) ++seen;
            else seen = t == tag.dsml ? 1 : 0;
            return;
          case 3:
            if (t == tag.eq_quote) { mode = Mode::kName; node = 0; seen = 0; ++spans; }
            else seen = t == tag.dsml ? 1 : 0;
            return;
          case 5:
            if (t == tag.eq_quote) { mode = Mode::kParam; seen = 0; }
            else seen = t == tag.dsml ? 1 : 0;
            return;
          default:
            seen = t == tag.dsml ? 1 : 0;
            return;
        }
    }
  }

  // The next row is constrained: the token after the emitted ones lies in a name span and the node has somewhere to go.
  bool in_span() const { return on && !in_think && mode == Mode::kName && node >= 0 && trie.first[node + 1] > trie.first[node]; }

  // The request's rule (temperature → top_k → top_p → min_p → multinomial) over the node's tokens only: the model's own distribution
  //   conditioned on the allowed tokens (temperature ≤ 0 = the best allowed token). Call only while in_span().
  int32_t sample(const float* row, int vocab, float temperature, float top_p, int top_k, float min_p, std::mt19937_64& rng) {
    ++rows;
    static thread_local std::vector<float> l;
    const int32_t b = trie.first[node], e = trie.first[node + 1];
    l.clear();
    for (int32_t i = b; i < e; ++i) l.push_back(trie.tok[i] < vocab ? row[trie.tok[i]] : -INFINITY);
    return trie.tok[b + sampling::sample(l, temperature, top_p, top_k, min_p, rng)];
  }

 private:
  int32_t edge(int32_t t) const {
    for (int32_t e = trie.first[node]; e < trie.first[node + 1]; ++e) if (trie.tok[e] == t) return e;
    return -1;
  }
};

}  // namespace hive::toolmask
