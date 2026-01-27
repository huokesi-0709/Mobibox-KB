"""
db_sqlitevec.py
负责：
- 加载 sqlite-vec 扩展（vec0）
- 建表（读取 sql/schema.sql）
- 插入 chunks 元数据 + 向量

关键点：
- schema.sql 路径不再依赖工作目录，改为从项目根目录 SQL_DIR 获取
"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite_vec  # pip install sqlite-vec

from monibox_kb.paths import SQL_DIR


def flat_pipe(items: List[str]) -> str:
    """
    把标签/人群数组扁平化成 |a|b| 的形式，
    便于 SQL 用 LIKE '%|tag|%' 精准匹配，避免子串误匹配。
    """
    items = [x.strip() for x in items if x and x.strip()]
    return "|" + "|".join(items) + "|"


class RagDB:
    def __init__(self, db_path: str, schema_path: Optional[Path] = None):
        self.db_path = db_path
        self.schema_path = schema_path or (SQL_DIR / "schema.sql")

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        sqlite_vec.load(conn)  # 加载 vec0 扩展
        return conn

    def create_tables(self):
        sql = Path(self.schema_path).read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(sql)

    def insert_chunks(self, records: List[Dict[str, Any]], vectors: List[List[float]]):
        """
        records 与 vectors 一一对应。
        vectors 写入 vec_chunks，rowid 对应 chunks.id
        """
        assert len(records) == len(vectors), "records 与 vectors 数量必须一致"

        with self.connect() as conn:
            cur = conn.cursor()
            for r, v in zip(records, vectors):
                cur.execute(
                    """
                    INSERT INTO chunks(
                      chunk_id, text, dimension, topic, risk,
                      source_id, status, quality_score, fingerprint,
                      tts_ok, tts_style, tags_flat, populations_flat
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["片段ID"],
                        r["文本"],
                        r["维度"],
                        r.get("子主题"),
                        r["风险等级"],
                        r["来源ID"],
                        r["状态"],
                        float(r.get("人工评分", 0)),
                        r["内容指纹"],
                        1 if r.get("可直接播报", True) else 0,
                        r.get("播报风格"),
                        flat_pipe(r.get("标签", [])),
                        flat_pipe(r.get("适用人群", [])),
                    )
                )
                rowid = cur.lastrowid
                cur.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, v))
            conn.commit()