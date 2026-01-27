"""
db_sqlitevec.py（写入 display_id / group_id 版）

- 启用 sqlite-vec 扩展加载
- 写 chunks 元数据（包含 display_id/group_id）
- 写 vec_chunks 向量（float32 BLOB）
"""

import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite_vec

from monibox_kb.paths import SQL_DIR


def flat_pipe(items: List[str]) -> str:
    items = [x.strip() for x in items if x and x.strip()]
    return "|" + "|".join(items) + "|"


def vec_to_f32_blob(vec: List[float]) -> sqlite3.Binary:
    floats = [float(x) for x in vec]
    blob = struct.pack("<%sf" % len(floats), *floats)
    return sqlite3.Binary(blob)


class RagDB:
    def __init__(self, db_path: str, schema_path: Optional[Path] = None):
        self.db_path = db_path
        self.schema_path = schema_path or (SQL_DIR / "schema.sql")

    def connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)

        return conn

    def create_tables(self):
        sql = Path(self.schema_path).read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(sql)

    def insert_chunks(self, records: List[Dict[str, Any]], vectors: List[List[float]]):
        assert len(records) == len(vectors), "records 与 vectors 数量必须一致"

        with self.connect() as conn:
            cur = conn.cursor()

            for r, v in zip(records, vectors):
                # 注意：display_id / group_id 可能缺失，允许为 None
                cur.execute(
                    """
                    INSERT INTO chunks(
                      chunk_id, display_id, group_id,
                      text, dimension, topic, risk,
                      source_id, status, quality_score, fingerprint,
                      tts_ok, tts_style,
                      tags_flat, populations_flat
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["片段ID"],
                        r.get("显示ID"),
                        r.get("片段组ID"),

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

                rowid = int(cur.lastrowid)
                blob = vec_to_f32_blob(v)

                cur.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )

            conn.commit()