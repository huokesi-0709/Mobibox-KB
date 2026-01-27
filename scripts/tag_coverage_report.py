"""
tag_coverage_report.py
用途：统计当前 rag.db 中 “标签/维度/状态/风险”等覆盖情况。

运行：
  python -m scripts.tag_coverage_report
  python -m scripts.tag_coverage_report --top 30
"""

import argparse
import sqlite3
import json
from collections import Counter, defaultdict
from typing import List, Dict, Set

from monibox_kb.config import settings
from monibox_kb.paths import GENERATED_DIR as GEN


def parse_tags_flat(tags_flat: str) -> List[str]:
    """
    tags_flat 形如：|tag1|tag2|
    解析成 ['tag1','tag2']
    """
    if not tags_flat:
        return []
    s = tags_flat.strip()
    if s == "|":
        return []
    # 去掉两端的 |
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return parts


def load_taxonomy_tag_ids() -> Set[str]:
    """
    读取 normalized taxonomy 的 tag_id 集合。
    若不存在，则返回空集合（不会报错）。
    """
    p = GEN / "02_meta_candidates.normalized.json"
    if not p.exists():
        return set()
    obj = json.loads(p.read_text(encoding="utf-8"))
    items = obj.get("标签体系", [])
    out = set()
    for it in items:
        tid = str(it.get("标签ID", "")).strip()
        if tid:
            out.add(tid)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=25, help="TopN 标签展示数量（默认25）")
    args = parser.parse_args()

    db_path = settings.rag_db_path
    print("==== tag_coverage_report ====")
    print("[info] db:", db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT
          chunk_id, display_id, group_id,
          dimension, risk, status, source_id,
          tags_flat, populations_flat
        FROM chunks
    """).fetchall()

    conn.close()

    total = len(rows)
    print("[info] total chunks:", total)
    if total == 0:
        print("DB is empty.")
        return

    # 统计维度/状态/风险
    dim_cnt = Counter()
    status_cnt = Counter()
    risk_cnt = Counter()
    source_cnt = Counter()

    # 标签统计
    tag_cnt = Counter()
    tag_dim_cnt = defaultdict(Counter)   # tag -> Counter(dimension -> count)

    for r in rows:
        dim = r["dimension"]
        status = r["status"]
        risk = r["risk"]
        source = r["source_id"]

        dim_cnt[dim] += 1
        status_cnt[status] += 1
        risk_cnt[risk] += 1
        source_cnt[source] += 1

        tags = parse_tags_flat(r["tags_flat"])
        for t in tags:
            tag_cnt[t] += 1
            tag_dim_cnt[t][dim] += 1

    print("\n-- dimension coverage --")
    for k, v in dim_cnt.most_common():
        print(f"{k}: {v}")

    print("\n-- status coverage --")
    for k, v in status_cnt.most_common():
        print(f"{k}: {v}")

    print("\n-- risk coverage --")
    for k, v in risk_cnt.most_common():
        print(f"{k}: {v}")

    print("\n-- source coverage --")
    for k, v in source_cnt.most_common():
        print(f"{k}: {v}")

    # Top tags
    print(f"\n-- top {args.top} tags --")
    for t, c in tag_cnt.most_common(args.top):
        # 显示该标签主要分布在哪些维度
        dim_top = tag_dim_cnt[t].most_common(3)
        dim_info = ", ".join([f"{d}:{n}" for d, n in dim_top])
        print(f"{t}: {c}   ({dim_info})")

    # taxonomy 对比
    tax_tags = load_taxonomy_tag_ids()
    if tax_tags:
        db_tags = set(tag_cnt.keys())

        missing_in_db = sorted(list(tax_tags - db_tags))
        unknown_in_tax = sorted(list(db_tags - tax_tags))

        print("\n-- taxonomy compare --")
        print("taxonomy tags:", len(tax_tags))
        print("db tags:", len(db_tags))

        print("\n[1] tags in taxonomy but NOT in db (coverage=0)  => 建议优先补数据：")
        if missing_in_db:
            print("count:", len(missing_in_db))
            print("sample:", missing_in_db[:50])
        else:
            print("none")

        print("\n[2] tags in db but NOT in taxonomy (可能是标签体系不一致/旧标签残留)：")
        if unknown_in_tax:
            print("count:", len(unknown_in_tax))
            print("sample:", unknown_in_tax[:50])
        else:
            print("none")
    else:
        print("\n[WARN] 未找到 02_meta_candidates.normalized.json，跳过 taxonomy 对比。")


if __name__ == "__main__":
    main()