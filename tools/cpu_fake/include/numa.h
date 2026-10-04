// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
// Fake libnuma for CPU-only tests. Reports two nodes (FAKE_NUMA_NODES=1 -> one) so the
// production two-node path runs; allocations are anonymous mmap; node cpus = all cpus.
#pragma once
#include <sys/mman.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
struct bitmask { unsigned long size; unsigned char* maskp; };
inline int numa_available() { return 0; }
inline int numa_max_node() { const char* s = getenv("FAKE_NUMA_NODES"); return s && atoi(s) == 1 ? 0 : 1; }
inline void* numa_alloc_onnode(size_t n, int) { void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); return p == MAP_FAILED ? nullptr : p; }
inline void numa_free(void* p, size_t n) { munmap(p, n); }
inline bitmask* numa_allocate_cpumask() { auto* b = new bitmask; b->size = (unsigned long)sysconf(_SC_NPROCESSORS_CONF); b->maskp = new unsigned char[b->size](); return b; }
inline int numa_node_to_cpus(int, bitmask* b) { memset(b->maskp, 1, b->size); return 0; }
inline int numa_bitmask_isbitset(const bitmask* b, unsigned int i) { return i < b->size && b->maskp[i]; }
inline void numa_free_cpumask(bitmask* b) { delete[] b->maskp; delete b; }
