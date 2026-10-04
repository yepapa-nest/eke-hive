#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Eke Hive Authors
"""Print the token ids from a golden directory's prefill.json, comma-separated (for use as a shell argument)."""
import json, sys
print(",".join(map(str, json.load(open(sys.argv[1] + "/prefill.json"))["ids"])))
