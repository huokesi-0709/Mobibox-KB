import argparse
import sqlite3
import struct
from typing import List, Optional, Dict
from collections import Counter

import sqlite_vec

from monibox_kb.config import settings
from monibox_kb.paths import PROJECT_ROOT
from monibox_kb.embedding import embed_texts
from monibox_kb.scoring.rerank import RerankPolicy, final_distance
from monibox_kb.routing.router import AutoRouter


def vec_to_f32_blob(vec: List[float]) -> sqlite3.Binary:
    floats = [float(x) for x in vec]
    blob = struct.pack("<%sf" % len(floats), *floats)
    return sqlite3.Binary(blob)


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def parse_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def diversify_by_group(scored_rows, topk: int, max_per_group: int = 1):
    """
    scored_rows: List[(final_distance, sqlite_row)]
    按 final_distance 排好序后，限制每个 group_id 最多出现 max_per_group 次。
    """
    picked = []
    group_cnt = Counter()

    for d_final, r in scored_rows:
        gid = r["group_id"] or ""
        if gid and group_cnt[gid] >= max_per_group:
            continue
        picked.append((d_final, r))
        if gid:
            group_cnt[gid] += 1
        if len(picked) >= topk:
            break

    # 如果去重后不够 topk，就补齐（忽略 group 限制）
    if len(picked) < topk:
        exists = set((r["chunk_id"] for _, r in picked))
        for d_final, r in scored_rows:
            if r["chunk_id"] in exists:
                continue
            picked.append((d_final, r))
            if len(picked) >= topk:
                break

    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="查询文本")
    parser.add_argument("--topk", type=int, default=5, help="返回条数")
    parser.add_argument("--dimension", default=None, help="限定维度（可选）")
    parser.add_argument("--risk", default=None, help="限定风险等级（可选，逗号分隔）")
    parser.add_argument("--tags", default=None, help="限定标签（可选，逗号分隔）")
    parser.add_argument("--status", default=None, help="限定状态（可选，逗号分隔），默认排除停用")
    parser.add_argument("--pool_mult", type=int, default=8, help="候选池倍率（默认8，用于评分重排）")

    parser.add_argument("--auto_route", action="store_true", help="自动推断 dimension/tags（若未手动指定）")
    parser.add_argument("--auto_top_tags", type=int, default=1, help="自动路由选取标签数量（默认1）")

    # 新增：多样性控制
    parser.add_argument("--max_per_group", type=int, default=1, help="同一 group_id 最多返回几条（默认1）")
    args = parser.parse_args()

    policy = RerankPolicy.load_default()

    db_path = settings.rag_db_path
    print("==== query_demo (rerank + auto_route + diversify) ====")
    print("[info] project root:", PROJECT_ROOT)
    print("[info] db path:", db_path)
    print("[info] query:", args.q)
    print("[info] policy:", policy)
    print()

    if args.auto_route and (args.tags is None or args.dimension is None):
        router = AutoRouter()
        rr = router.route(args.q, top_tags=args.auto_top_tags)

        print("[route] predicted dimension:", rr.dimension)
        print("[route] predicted tags:", rr.tags)
        print("[route] tag_dims:", rr.tag_dims)
        print("[route] cross_dimension:", rr.cross_dimension)
        print("[route] evidence:", rr.evidence)
        print()

        if args.tags is None and rr.tags:
            args.tags = ",".join(rr.tags)

        if args.dimension is None:
            if rr.cross_dimension:
                print("[route] cross-dimension detected -> UNLOCK dimension filter (dimension=None)\n")
                args.dimension = None
            else:
                args.dimension = rr.dimension

    qvec = embed_texts([args.q])[0]
    qblob = vec_to_f32_blob(qvec)

    conn = open_db(db_path)

    where = []
    params: Dict[str, object] = {}

    if args.status:
        statuses = parse_csv(args.status)
        keys = []
        for i, st in enumerate(statuses):
            k = f"s{i}"
            params[k] = st
            keys.append(f":{k}")
        where.append(f"c.status IN ({','.join(keys)})")
    else:
        where.append("c.status <> '停用'")

    if args.dimension:
        where.append("c.dimension = :dimension")
        params["dimension"] = args.dimension

    if args.risk:
        risks = parse_csv(args.risk)
        keys = []
        for i, rv in enumerate(risks):
            k = f"r{i}"
            params[k] = rv
            keys.append(f":{k}")
        where.append(f"c.risk IN ({','.join(keys)})")

    # tags OR
    if args.tags:
        tags = parse_csv(args.tags)
        ors = []
        for i, t in enumerate(tags):
            k = f"t{i}"
            params[k] = f"%|{t}|%"
            ors.append(f"c.tags_flat LIKE :{k}")
        where.append("(" + " OR ".join(ors) + ")")

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    topk = int(args.topk)
    k_pool = min(max(topk, topk * int(args.pool_mult)), 300)

    sql = f"""
    WITH knn AS (
      SELECT rowid, distance
      FROM vec_chunks
      WHERE embedding MATCH :qvec
        AND k = :kpool
    )
    SELECT
      c.chunk_id,
      c.display_id,
      c.group_id,
      c.text,
      c.dimension,
      c.risk,
      c.source_id,
      c.status,
      c.quality_score,
      knn.distance
    FROM knn
    JOIN chunks c ON c.id = knn.rowid
    {where_sql}
    ORDER BY knn.distance
    LIMIT :kpool;
    """

    params["qvec"] = qblob
    params["kpool"] = int(k_pool)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        print("No results.")
        return

    # rerank
    scored = []
    for r in rows:
        d = float(r["distance"])
        q = float(r["quality_score"])
        st = r["status"]
        d_final = final_distance(d, q, st, policy)
        scored.append((d_final, r))

    scored.sort(key=lambda x: x[0])

    # diversify by group_id
    final = diversify_by_group(scored, topk=topk, max_per_group=max(1, int(args.max_per_group)))

    print(f"Candidates={len(rows)}  ReturnTopK={len(final)}  max_per_group={args.max_per_group}")
    for i, (d_final, r) in enumerate(final, start=1):
        text = r["text"]
        if len(text) > 120:
            text = text[:120] + "..."
        print(f"\n[{i}] chunk_id={r['chunk_id']}")
        print(f"    display_id={r['display_id']}  group_id={r['group_id']}")
        print(f"    distance={float(r['distance']):.6f}  final_distance={d_final:.6f}  score={float(r['quality_score']):.1f}  status={r['status']}")
        print(f"    dimension={r['dimension']}  risk={r['risk']}  source={r['source_id']}")
        print(f"    text={text}")


if __name__ == "__main__":
    main()