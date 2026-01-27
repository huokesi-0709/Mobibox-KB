"""
rate_chunk.py
用途：测试闭环工具（给某条片段打分/停用/启用）

用法示例：
1) 通过显示ID打分
   python -m scripts.rate_chunk --display_id "k_synth_psy_panic_9106c91e_00" --score 5

2) 通过片段ID停用
   python -m scripts.rate_chunk --chunk_id "k_qa_00012_xxxxxx_00_13005e64" --status 停用

3) 查询某个显示ID对应的记录
   python -m scripts.rate_chunk --display_id "k_synth_psy_panic_9106c91e_00" --show
"""

import argparse
import sqlite3
from monibox_kb.config import settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", default=None, help="按片段ID定位（chunk_id）")
    parser.add_argument("--display_id", default=None, help="按显示ID定位（display_id）")
    parser.add_argument("--score", type=float, default=None, help="设置评分 quality_score（0~5建议）")
    parser.add_argument("--status", default=None, help="设置状态：候选/启用/停用")
    parser.add_argument("--show", action="store_true", help="只查看，不修改")
    args = parser.parse_args()

    if not args.chunk_id and not args.display_id:
        raise ValueError("必须提供 --chunk_id 或 --display_id 之一")

    db_path = settings.rag_db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 定位记录
    if args.chunk_id:
        rows = cur.execute("SELECT * FROM chunks WHERE chunk_id = ?", (args.chunk_id,)).fetchall()
    else:
        rows = cur.execute("SELECT * FROM chunks WHERE display_id = ?", (args.display_id,)).fetchall()

    if not rows:
        conn.close()
        print("Not found.")
        return

    # 展示
    def show_row(r):
        print("chunk_id:", r["chunk_id"])
        print("display_id:", r["display_id"])
        print("group_id:", r["group_id"])
        print("dimension:", r["dimension"], "risk:", r["risk"], "status:", r["status"], "source:", r["source_id"])
        print("quality_score:", r["quality_score"])
        print("text:", r["text"])
        print("-" * 60)

    for r in rows:
        show_row(r)

    if args.show:
        conn.close()
        return

    # 修改
    sets = []
    params = []

    if args.score is not None:
        sets.append("quality_score = ?")
        params.append(float(args.score))

    if args.status is not None:
        if args.status not in ("候选", "启用", "停用"):
            conn.close()
            raise ValueError("status 只能是：候选/启用/停用")
        sets.append("status = ?")
        params.append(args.status)

    if not sets:
        conn.close()
        print("No changes. (use --score or --status)")
        return

    if args.chunk_id:
        where = "chunk_id = ?"
        params.append(args.chunk_id)
    else:
        where = "display_id = ?"
        params.append(args.display_id)

    sql = f"UPDATE chunks SET {', '.join(sets)} WHERE {where}"
    cur.execute(sql, params)
    conn.commit()

    print("UPDATED rows:", cur.rowcount)

    # 再次展示
    if args.chunk_id:
        r2 = cur.execute("SELECT * FROM chunks WHERE chunk_id = ?", (args.chunk_id,)).fetchone()
    else:
        r2 = cur.execute("SELECT * FROM chunks WHERE display_id = ?", (args.display_id,)).fetchone()

    if r2:
        show_row(r2)

    conn.close()


if __name__ == "__main__":
    main()