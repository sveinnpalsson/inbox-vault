#!/usr/bin/env python3
"""Lightweight no-dependency contract validation for core flow."""

from inbox_vault.llm import extract_first_json


def main() -> int:
    sample = 'x {"category":"Important","importance":8,"action":"reply","summary":"foo"} y'
    parsed = extract_first_json(sample)
    if not parsed:
        print("FAIL: could not parse JSON")
        return 1
    required = {"category", "importance", "action", "summary"}
    if not required.issubset(parsed.keys()):
        print(f"FAIL: missing keys: {required - set(parsed.keys())}")
        return 1
    print("OK: contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
