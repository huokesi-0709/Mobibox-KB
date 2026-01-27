import argparse
import sqlite3
import struct
from typing import List, Optional

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="查询文本")
    parser.add_argument("--topk", type=int, default=5, help="返回条数")
    parser.add_argument("--dimension", default=None, help="限定维度（可选）")
    parser.add_argument("--risk", default=None, help="限定风险等级（可选，逗号分隔）")
    parser.add_argument("--tags", default=None, help="必须包含的标签ID（可选，逗号分隔）")
    parser.add_argument("--status", default=None, help="限定状态（可选，逗号分隔），默认排除停用")
    parser.add_argument("--pool_mult", type=int, default=8, help="候选池倍率（默认8，用于评分重排）")

    # 新增：自动路由
    parser.add_argument("--auto_route", action="store_true", help="自动推断 dimension/tags（若未手动指定）")
    parser.add_argument("--auto_top_tags", type=int, default=1, help="自动路由选取的标签数量（默认1）")

    args = parser.parse_args()

    policy = RerankPolicy.load_default()

    db_path = settings.rag_db_path
    print("==== query_demo (rerank + auto_route) ====")
    print("[info] project root:", PROJECT_ROOT)
    print("[info] db path:", db_path)
    print("[info] query:", args.q)
    print("[info] policy:", policy)
    print()

    # 1) auto route（仅当用户没手动指定 dimension/tags 时）
    if args.auto_route and (args.dimension is None or args.tags is None):
        router = AutoRouter()
        rr = router.route(args.q, top_tags=args.auto_top_tags)
        print("[route] predicted dimension:", rr.dimension)
        print("[route] predicted tags:", rr.tags)
        print("[route] evidence:", rr.evidence)
        print()

        if args.dimension is None:
            args.dimension = rr.dimension
        if args.tags is None and rr.tags:
            # 只用路由出的标签做过滤
            args.tags = ",".join(rr.tags)

    # 2) embedding
    qvec = embed_texts([args.q])[0]
    qblob = vec_to_f32_blob(qvec)

    # 3) open db
    conn = open_db(db_path)

    where = []
    params = {}

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

    # 注意：这里 tags 用 OR 逻辑（任意命中一个标签即可），更适合路由结果
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
    final = scored[:topk]

    print(f"Candidates={len(rows)}  ReturnTopK={len(final)}")
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