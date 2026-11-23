#!/usr/bin/env python3
"""Score one completion against a gold answer in a separate process.

Reads {"completion": ..., "answer": ...} as JSON on stdin and prints 1 or 0.
Model output is untrusted (very long numbers, pathological text), so the trainer runs
this with a timeout instead of parsing in-process.
"""
import json
import sys

from common import extract

item = json.load(sys.stdin)
print(1 if extract(item["completion"]) == item["answer"] else 0)
