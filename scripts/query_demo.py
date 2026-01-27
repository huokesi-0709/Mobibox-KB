"""
query_demo.py（评分重排版）

关键修复：
- sqlite-vec 的 vec0 KNN 查询在某些情况下（尤其 JOIN 时）要求显式提供 k 约束：
  WHERE embedding MATCH :qvec AND k = :k
- 所以我们把 KNN 搜索放到 CTE 子查询里，再 JOIN chunks 表做过滤/展示。

运行：
  python -m scripts.query_demo --q "我好害怕，喘不过气" --topk 5
  python -m scripts.query_demo --q "我腿被压住了，发麻动不了" --dimension "核心生理病理学" --topk 8
  python -m scripts.query_demo --q "又在晃，是不是余震" --tags scn_aftershock --topk 5

参数说明：
- --q            必填：查询文本
- --topk         返回多少条
- --dimension    可选：限定维度
- --risk         可选：限定风险等级（逗号分隔，如 "中,高"）
- --tags         可选：必须包含的标签ID（逗号分隔，如 "psy_panic,act_breath_pacing"）
- --status       可选：限定状态（逗号分隔），默认排除“停用”
"""



import argparse
import sqlite3
import struct
from typing import List, Optional

import sqlite_vec

from monibox_kb.config import settings
from monibox_kb.paths import PROJECT_ROOT
from monibox_kb.embedding import embed_texts
from monibox_kb.scoring.rerank import RerankPolicy, final_distance


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
    parser.add_argument("--pool_mult", type=int, default=8, help="候选池倍率（用于重排），默认8")
    args = parser.parse_args()

    policy = RerankPolicy.load_default()

    db_path = settings.rag_db_path
    print("==== query_demo (rerank) ====")
    print("[info] project root:", PROJECT_ROOT)
    print("[info] db path:", db_path)
    print("[info] query:", args.q)
    print("[info] policy:", policy)
    print()

    # 1) embedding
    qvec = embed_texts([args.q])[0]
    qblob = vec_to_f32_blob(qvec)

    # 2) open db
    conn = open_db(db_path)

    # 3) filters on chunks
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

    if args.tags:
        tags = parse_csv(args.tags)
        for i, t in enumerate(tags):
            k = f"t{i}"
            params[k] = f"%|{t}|%"
            where.append(f"c.tags_flat LIKE :{k}")

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    # 4) candidate pool
    topk = int(args.topk)
    k_pool = max(topk, topk * int(args.pool_mult))
    # 防止太大（调试用）
    k_pool = min(k_pool, 300)

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

    # 5) rerank in python
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