from __future__ import annotations

import json
import math
from pathlib import Path

from .config import AppConfig
from .vectors import search_vectors


def _load_eval_cases(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("Eval file must be a JSON array")
    cases: list[dict] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid eval case at index {idx}: expected object")
        query = str(item.get("query", "")).strip()
        if not query:
            raise ValueError(f"Invalid eval case at index {idx}: missing query")
        relevant = item.get("relevant_msg_ids", [])
        if not isinstance(relevant, list):
            raise ValueError(f"Invalid eval case at index {idx}: relevant_msg_ids must be array")
        cases.append(
            {
                "query": query,
                "relevant_msg_ids": [str(v) for v in relevant],
                "account_email": item.get("account_email"),
                "label": item.get("label"),
                "scope": item.get("scope"),
            }
        )
    return cases


def _dcg_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    score = 0.0
    for idx, msg_id in enumerate(predicted_ids[:k], start=1):
        if msg_id in relevant_ids:
            score += 1.0 / math.log2(idx + 1)
    return score


def _ndcg_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = _dcg_at_k(predicted_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _slice_key(case: dict) -> str | None:
    if case.get("label"):
        return f"label:{str(case['label']).strip().upper()}"
    if case.get("scope"):
        return f"scope:{str(case['scope']).strip()}"
    if case.get("account_email"):
        return f"account:{str(case['account_email']).strip().lower()}"
    return None


def _metric_summary(items: list[dict], *, strategy: str, top_k: int) -> dict:
    total = len(items)
    if total == 0:
        return {
            "cases": 0,
            "strategy": strategy,
            "top_k": top_k,
            "recall_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "hits": 0,
        }

    hits = sum(1 for item in items if item["first_relevant_rank"] is not None)
    rr_sum = sum(item["reciprocal_rank"] for item in items)
    ndcg_sum = sum(item["ndcg_at_k"] for item in items)

    return {
        "cases": total,
        "strategy": strategy,
        "top_k": top_k,
        "recall_at_k": round(hits / total, 6),
        "mrr_at_k": round(rr_sum / total, 6),
        "ndcg_at_k": round(ndcg_sum / total, 6),
        "hits": hits,
    }


def run_retrieval_eval(
    conn,
    cfg: AppConfig,
    *,
    eval_file: str,
    strategy: str,
    top_k: int,
    clearance: str = "redacted",
) -> dict:
    cases = _load_eval_cases(eval_file)

    if not cases:
        return _metric_summary([], strategy=strategy, top_k=top_k)

    details: list[dict] = []

    for case in cases:
        results = search_vectors(
            conn,
            cfg,
            case["query"],
            account_email=case.get("account_email"),
            label=case.get("label"),
            top_k=top_k,
            clearance=clearance,
            strategy=strategy,
        )
        predicted_ids = [item.msg_id for item in results]
        relevant_ids = set(case["relevant_msg_ids"])

        first_rank = None
        for idx, msg_id in enumerate(predicted_ids, start=1):
            if msg_id in relevant_ids:
                first_rank = idx
                break

        reciprocal_rank = 0.0 if first_rank is None else (1.0 / first_rank)
        ndcg = _ndcg_at_k(predicted_ids, relevant_ids, top_k)

        details.append(
            {
                "query": case["query"],
                "relevant_msg_ids": sorted(relevant_ids),
                "predicted_msg_ids": predicted_ids,
                "first_relevant_rank": first_rank,
                "reciprocal_rank": round(reciprocal_rank, 6),
                "ndcg_at_k": round(ndcg, 6),
                "label": case.get("label"),
                "scope": case.get("scope"),
                "account_email": case.get("account_email"),
            }
        )

    summary = _metric_summary(details, strategy=strategy, top_k=top_k)

    slices: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = {}
    for detail, case in zip(details, cases):
        key = _slice_key(case)
        if not key:
            continue
        grouped.setdefault(key, []).append(detail)

    for key in sorted(grouped):
        slices[key] = _metric_summary(grouped[key], strategy=strategy, top_k=top_k)

    return {
        **summary,
        "details": details,
        "slices": slices,
    }


def bootstrap_eval_template(
    conn,
    *,
    output_file: str,
    limit: int = 20,
) -> dict:
    safe_limit = max(1, int(limit))
    rows = conn.execute(
        """
        SELECT msg_id, account_email, labels_json, date_iso
        FROM messages
        ORDER BY COALESCE(internal_ts, 0) DESC
        LIMIT ?
        """,
        (safe_limit,),
    ).fetchall()

    cases: list[dict] = []
    for idx, (msg_id, account_email, labels_json, date_iso) in enumerate(rows, start=1):
        labels = [str(v).upper() for v in (json.loads(labels_json or "[]") if labels_json else [])]
        preferred_label = "INBOX" if "INBOX" in labels else (labels[0] if labels else None)
        case = {
            "query": f"TODO: add retrieval query #{idx}",
            "relevant_msg_ids": [msg_id],
            "account_email": account_email,
            "label": preferred_label,
            "scope": "starter",
            "meta": {
                "source": "bootstrap_eval_template",
                "date_iso": date_iso,
                "labels": labels,
            },
        }
        cases.append(case)

    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, indent=2))

    return {
        "written": len(cases),
        "output_file": str(output_path),
        "note": "Edit TODO queries before running eval-retrieval",
    }
