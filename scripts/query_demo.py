"""
query_demo.py（修复版）

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


def vec_to_f32_blob(vec: List[float]) -> sqlite3.Binary:
    """
    将 512维向量 list[float] 转成 float32 little-endian BLOB。
    必须与入库时 vec0 接受的格式一致。
    """
    floats = [float(x) for x in vec]
    blob = struct.pack("<%sf" % len(floats), *floats)
    return sqlite3.Binary(blob)


def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # sqlite-vec 扩展需要 enable_load_extension(True)
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
    parser.add_argument("--topk", type=int, default=5, help="返回条数（同时作为 vec0 的 k）")
    parser.add_argument("--dimension", default=None, help="限定维度（可选）")
    parser.add_argument("--risk", default=None, help="限定风险等级，逗号分隔（可选）")
    parser.add_argument("--tags", default=None, help="必须包含的标签ID，逗号分隔（可选）")
    parser.add_argument("--status", default=None, help="限定状态，逗号分隔（可选），默认排除停用")

    args = parser.parse_args()

    db_path = settings.rag_db_path
    print("==== query_demo ====")
    print("[info] project root:", PROJECT_ROOT)
    print("[info] db path:", db_path)
    print("[info] query:", args.q)
    print()

    # 1) embedding 查询
    qvec = embed_texts([args.q])[0]
    qblob = vec_to_f32_blob(qvec)

    # 2) 连接 DB
    conn = open_db(db_path)

    # 3) 组装 SQL 过滤条件（作用在 chunks 表）
    where = []
    params = {}

    # 默认排除停用
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

    # 4) 关键：先在 vec_chunks 内做 KNN，并显式给 k
    # 再 JOIN chunks 做过滤/展示
    sql = f"""
    WITH knn AS (
      SELECT rowid, distance
      FROM vec_chunks
      WHERE embedding MATCH :qvec
        AND k = :k
    )
    SELECT
      c.chunk_id,
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
    LIMIT :topk;
    """

    params["qvec"] = qblob
    params["k"] = int(args.topk)       # vec0 必需的 k
    params["topk"] = int(args.topk)    # 输出条数（保持一致）

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    # 5) 输出结果
    if not rows:
        print("No results.")
        return

    print(f"Top {len(rows)} results:")
    for i, r in enumerate(rows, start=1):
        text = r["text"]
        if len(text) > 120:
            text = text[:120] + "..."
        print(f"\n[{i}] chunk_id={r['chunk_id']}")
        print(f"    distance={r['distance']:.6f}")
        print(f"    dimension={r['dimension']}  risk={r['risk']}  status={r['status']}  source={r['source_id']}  score={r['quality_score']}")
        print(f"    text={text}")


if __name__ == "__main__":
    main()