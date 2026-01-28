"""
auto_flag_unsafe_chunks.py (v2)
用途：
- 扫描 rag.db chunks.text
- 用 SafetyGuard 判断：allow / rewrite / block
- 输出审计报告
- 可选对：
  - block：停用（status='停用'）
  - rewrite：降权（quality_score 设为负数）或停用

推荐运行：
1) 只预览：
   python -m scripts.auto_flag_unsafe_chunks --dry_run

2) 实际执行（默认：block停用，rewrite仅报告不处理）：
   python -m scripts.auto_flag_unsafe_chunks

3) 实际执行（block停用 + rewrite降权到-1，推荐）：
   python -m scripts.auto_flag_unsafe_chunks --rewrite_action penalize --rewrite_penalty -1

4) 实际执行（block停用 + rewrite也停用，最保守）：
   python -m scripts.auto_flag_unsafe_chunks --rewrite_action disable
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any

from monibox_kb.config import settings
from monibox_kb.paths import GENERATED_DIR as GEN
from monibox_kb.runtime.safety_guard import SafetyGuard


REPORT_PATH = GEN / "unsafe_chunks_report.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="只预览，不更新数据库")
    ap.add_argument("--limit", type=int, default=0, help="最多扫描多少条（0表示全量）")

    ap.add_argument("--rewrite_action", choices=["report", "penalize", "disable"], default="report",
                    help="对 rewrite 级别内容的处理方式：report/penalize/disable")
    ap.add_argument("--rewrite_penalty", type=float, default=-1.0,
                    help="rewrite_action=penalize 时，将 quality_score 降到该值（默认 -1）")

    args = ap.parse_args()

    db_path = settings.rag_db_path
    guard = SafetyGuard()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, chunk_id, display_id, group_id, text, status, source_id, dimension, risk, quality_score
        FROM chunks
    """).fetchall()

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    block_items: List[Dict[str, Any]] = []
    rewrite_items: List[Dict[str, Any]] = []
    allow_count = 0

    to_disable_block: List[int] = []
    to_disable_rewrite: List[int] = []
    to_penalize_rewrite: List[tuple] = []   # (new_score, id)

    for r in rows:
        text = r["text"] or ""
        res = guard.check(text)

        base = {
            "id": int(r["id"]),
            "chunk_id": r["chunk_id"],
            "display_id": r["display_id"],
            "group_id": r["group_id"],
            "dimension": r["dimension"],
            "risk": r["risk"],
            "source_id": r["source_id"],
            "status_before": r["status"],
            "quality_score_before": r["quality_score"],
            "reasons": res.reasons,
            "text": text
        }

        if res.level == "block":
            block_items.append(base)
            to_disable_block.append(int(r["id"]))

        elif res.level == "rewrite":
            rewrite_items.append(base)

            if args.rewrite_action == "disable":
                to_disable_rewrite.append(int(r["id"]))

            elif args.rewrite_action == "penalize":
                # 只对未启用的条目降权，避免覆盖人工启用与高分
                if r["status"] != "启用":
                    cur_score = float(r["quality_score"] or 0.0)
                    penalty = float(args.rewrite_penalty)
                    if cur_score > penalty:
                        to_penalize_rewrite.append((penalty, int(r["id"])))

        else:
            allow_count += 1

    # 写报告（无论 dry_run 与否都写，便于审计）
    REPORT_PATH.write_text(json.dumps({
        "db": db_path,
        "scanned": len(rows),
        "allow": allow_count,
        "rewrite": len(rewrite_items),
        "block": len(block_items),
        "rewrite_action": args.rewrite_action,
        "rewrite_penalty": args.rewrite_penalty,
        "items": {
            "block": block_items,
            "rewrite": rewrite_items
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("==== auto_flag_unsafe_chunks (v2) ====")
    print("db:", db_path)
    print("scanned:", len(rows))
    print("allow:", allow_count)
    print("rewrite:", len(rewrite_items))
    print("block:", len(block_items))
    print("report:", REPORT_PATH)

    if args.dry_run:
        print("[dry_run] no db update.")
        conn.close()
        return

    # 执行更新
    updated = 0

    if to_disable_block:
        cur.executemany("UPDATE chunks SET status='停用' WHERE id=?", [(i,) for i in to_disable_block])
        updated += cur.rowcount

    if args.rewrite_action == "disable" and to_disable_rewrite:
        cur.executemany("UPDATE chunks SET status='停用' WHERE id=?", [(i,) for i in to_disable_rewrite])
        updated += cur.rowcount

    if args.rewrite_action == "penalize" and to_penalize_rewrite:
        cur.executemany("UPDATE chunks SET quality_score=? WHERE id=?", to_penalize_rewrite)
        updated += cur.rowcount

    conn.commit()
    conn.close()
    print("UPDATED rows:", updated)


if __name__ == "__main__":
    main()