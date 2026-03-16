#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from inbox_vault.config import load_config, resolve_password
from inbox_vault.db import get_conn


def main() -> None:
    parser = argparse.ArgumentParser(description="Print persisted redaction entries from SQLCipher DB")
    parser.add_argument("--config", default=None, help="Path to config TOML (default: ./config.toml)")
    parser.add_argument("--scope-type", default="account")
    parser.add_argument("--scope-id", default=None, help="Account email / scope id filter")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    password = resolve_password(cfg.db)
    conn = get_conn(cfg.db.path, password)
    try:
        sql = (
            "SELECT scope_type, scope_id, key_name, placeholder, value_norm, original_value, "
            "source_mode, hit_count, first_seen_at, last_seen_at "
            "FROM redaction_entries "
        )
        params: list[object] = []
        clauses: list[str] = []

        if args.scope_type:
            clauses.append("scope_type = ?")
            params.append(args.scope_type)
        if args.scope_id:
            clauses.append("scope_id = ?")
            params.append(args.scope_id)

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)

        sql += " ORDER BY scope_id, key_name, placeholder LIMIT ?"
        params.append(max(1, int(args.limit)))

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            print("No redaction_entries rows found for the requested scope/filter.")
            return

        print(
            "scope_type\tscope_id\tkey_name\tplaceholder\tvalue_norm\toriginal_value\tsource_mode\thit_count"
        )
        for row in rows:
            print(
                f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{row[5]}\t{row[6]}\t{row[7]}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
