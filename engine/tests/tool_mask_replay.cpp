// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Replays a token sequence through hive/tool_mask.h (tools/test_tool_calls_ref.py: real tokenizer, real rendered tool calls).
//   stdin, whitespace separated: dsml calls invoke parameter name eq · think_start think_end in_think · n first[n] · n tok[n] · n next[n] · n ids[n]
//   stdout: spans left mode in_think · then one line per name span: the tokens fed inside it.
#include <cstdio>
#include <iostream>
#include <vector>

#include "hive/tool_mask.h"

static std::vector<int32_t> vec() { int n; std::cin >> n; std::vector<int32_t> v(n); for (auto& x : v) std::cin >> x; return v; }

int main() {
  hive::toolmask::State s;
  s.on = true;
  std::cin >> s.tag.dsml >> s.tag.calls >> s.tag.invoke >> s.tag.parameter >> s.tag.name >> s.tag.eq_quote;
  int think;
  std::cin >> s.think_start >> s.think_end >> think;
  s.in_think = think != 0;
  s.trie.first = vec(); s.trie.tok = vec(); s.trie.next = vec();
  if (!s.trie.valid(1 << 30)) { puts("invalid"); return 2; }
  const std::vector<int32_t> ids = vec();
  std::vector<std::vector<int32_t>> spans;
  for (int32_t t : ids) {
    const bool in = s.mode == hive::toolmask::Mode::kName && !s.in_think;
    if (in) spans.back().push_back(t);
    const long before = s.spans;
    s.feed(t);
    if (s.spans != before) spans.emplace_back();
  }
  printf("%ld %ld %d %d\n", s.spans, s.left, (int)s.mode, (int)s.in_think);
  for (const auto& v : spans) { for (int32_t t : v) printf("%d ", t); puts(""); }
  return 0;
}
