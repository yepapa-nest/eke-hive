// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// hive/tool_mask.h on the CPU: the block / parameter / name-span state machine on the reference grammar's token sequence, the thinking span, a value quoting an
//   invoke tag, a second call in one block, the automaton's validation, and the constrained sampler: only the node's tokens come out, and their frequencies are
//   the unconstrained rule's renormalized over them. Negative control: the unconstrained sampler does draw other tokens on the same rows.
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "hive/tool_mask.h"

using namespace hive::toolmask;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++fails; } } while (0)

enum { DSML = 100, CALLS, INVOKE, PARAM, NAME, EQ, LT, LTS, GT, CLOSE, THINK, ENDTHINK, A, B, C, X };

static State fresh() {
  State s; s.on = true;
  s.tag = {DSML, CALLS, INVOKE, PARAM, NAME, EQ};
  s.think_start = THINK; s.think_end = ENDTHINK;
  // node 0: A → node 1, B → closes · node 1: C → closes
  s.trie.first = {0, 2, 3}; s.trie.tok = {A, B, C}; s.trie.next = {1, -1, -1};
  return s;
}
static void feed(State& s, std::initializer_list<int> ts) { for (int t : ts) s.feed(t); }

int main() {
  {  // a block with two calls, a parameter between them
    State s = fresh();
    feed(s, {X, LT, DSML, INVOKE, NAME, EQ});  EXPECT(!s.in_span(), "an invoke tag outside a block is text");
    feed(s, {GT, LT, DSML, CALLS, GT});        EXPECT(!s.in_span(), "block opened");
    feed(s, {LT, DSML, INVOKE, NAME});         EXPECT(!s.in_span(), "name attribute not open yet");
    s.feed(EQ);                                EXPECT(s.in_span() && s.node == 0 && s.spans == 1, "name span");
    s.feed(A);                                 EXPECT(s.in_span() && s.node == 1, "inside the name");
    s.feed(C);                                 EXPECT(!s.in_span() && s.mode == Mode::kBlock, "name closed");
    feed(s, {LT, DSML, PARAM, NAME, EQ, X, GT, X, X}); EXPECT(!s.in_span() && s.mode == Mode::kParam, "in a value");
    feed(s, {LT, DSML, INVOKE, NAME, EQ});     EXPECT(!s.in_span() && s.spans == 1, "a value quoting an invoke tag is not a span");
    feed(s, {LTS, DSML, PARAM, GT});           EXPECT(s.mode == Mode::kBlock, "value closed");
    feed(s, {LTS, DSML, INVOKE, GT});          EXPECT(!s.in_span(), "closing invoke has no name");
    feed(s, {LT, DSML, INVOKE, NAME, EQ});     EXPECT(s.in_span() && s.spans == 2, "second call");
    s.feed(B);                                 EXPECT(!s.in_span(), "closed by one token");
    feed(s, {LTS, DSML, INVOKE, GT, LTS, DSML, CALLS, GT}); EXPECT(s.mode == Mode::kText, "block closed");
    feed(s, {LT, DSML, INVOKE, NAME, EQ});     EXPECT(!s.in_span() && s.left == 0, "text again");
  }
  {  // thinking: tags written in the reasoning are not a block; a token off the trie leaves the span and is counted
    State s = fresh();
    s.in_think = true;
    feed(s, {LT, DSML, CALLS, GT, LT, DSML, INVOKE, NAME, EQ}); EXPECT(!s.in_span() && s.mode == Mode::kText, "thinking is free text");
    s.feed(ENDTHINK); EXPECT(!s.in_think, "</think> closes");
    feed(s, {LT, DSML, CALLS, GT, LT, DSML, INVOKE, NAME, EQ}); EXPECT(s.in_span(), "answer span");
    s.feed(X); EXPECT(!s.in_span() && s.left == 1, "off the trie: leaves the span, counted");
    State t = fresh();
    feed(t, {THINK, LT, DSML, CALLS}); EXPECT(t.in_think && t.mode == Mode::kText, "<think> opens in text");
    State o; o.tag = {DSML, CALLS, INVOKE, PARAM, NAME, EQ}; o.feed(DSML); o.feed(CALLS); EXPECT(!o.in_span() && o.mode == Mode::kText, "off = nothing tracked");
  }
  {  // validation
    Trie t; t.first = {0, 1}; t.tok = {5}; t.next = {-1};
    EXPECT(t.valid(10), "valid");
    Trie a = t; a.tok = {10};   EXPECT(!a.valid(10), "token outside the vocabulary");
    Trie b = t; b.next = {1};   EXPECT(!b.valid(10), "next past the last node");
    Trie d = t; d.next = {-2};  EXPECT(!d.valid(10), "next below −1");
    Trie e = t; e.first = {0, 2}; EXPECT(!e.valid(10), "first past the edges");
    Trie f; EXPECT(!f.valid(10), "empty");
    Trie g; g.first = {0, 2, 1, 3}; g.tok = {1, 2, 3}; g.next = {-1, -1, -1}; EXPECT(!g.valid(10), "first not monotone");
    Trie h; h.first = {1, 1}; h.tok = {5}; h.next = {-1}; EXPECT(!h.valid(10), "first does not start at 0");
  }
  {  // sampler: only the node's tokens; greedy = the best of them; frequencies = the renormalized rule
    const int V = 64;
    std::vector<float> row(V, 0.f);
    std::mt19937_64 rng(7);
    for (int i = 0; i < V; ++i) row[i] = 0.1f * (i % 7);
    row[9] = 5.f;       // the model's favourite, not allowed
    row[3] = 1.f; row[11] = 2.f; row[20] = 0.5f;
    State s = fresh();
    s.trie.first = {0, 3}; s.trie.tok = {3, 11, 20}; s.trie.next = {-1, -1, -1};
    s.mode = Mode::kName; s.node = 0;
    EXPECT(s.sample(row.data(), V, 0.f, 1.f, 0, 0.f, rng) == 11, "greedy = best allowed");
    std::map<int, int> n;
    const int N = 200000;
    for (int i = 0; i < N; ++i) ++n[s.sample(row.data(), V, 1.f, 1.f, 0, 0.f, rng)];
    EXPECT(n.size() == 3 && n.count(9) == 0, "only the node's tokens");
    const double e3 = std::exp(1.0), e11 = std::exp(2.0), e20 = std::exp(0.5), tot = e3 + e11 + e20;
    auto close = [&](int t, double p) { const double se = std::sqrt(p * (1 - p) / N); return std::fabs(n[t] / (double)N - p) < 5 * se; };
    EXPECT(close(3, e3 / tot) && close(11, e11 / tot) && close(20, e20 / tot), "frequencies = the model's, renormalized");
    EXPECT(s.rows == N + 1, "rows counted");
    // top_p over the allowed tokens: 0.5 keeps only the best
    for (int i = 0; i < 200; ++i) EXPECT(s.sample(row.data(), V, 1.f, 0.5f, 0, 0.f, rng) == 11, "top_p applies to the allowed set");
    // a token past the row (a trie built for a larger vocabulary) is never chosen
    s.trie.tok = {3, 11, 63}; row[63] = 9.f;
    EXPECT(s.sample(row.data(), 40, 0.f, 1.f, 0, 0.f, rng) == 11, "token outside the row is out");
    // negative control: the unconstrained rule draws other tokens
    std::vector<float> l = row; bool other = false;
    for (int i = 0; i < 2000; ++i) { const int t = hive::sampling::sample(l, 1.f, 1.f, 0, 0.f, rng); other |= t != 3 && t != 11 && t != 20; }
    EXPECT(other, "unconstrained sampler reaches other tokens");
    // a node without edges does not constrain
    State z = fresh(); z.trie.first = {0, 0}; z.trie.tok.clear(); z.trie.next.clear(); z.mode = Mode::kName; z.node = 0;
    EXPECT(!z.in_span(), "no edges: not constrained");
  }
  if (fails) { fprintf(stderr, "tool mask: %d failures\n", fails); return 1; }
  puts("tool mask: ok");
  return 0;
}
