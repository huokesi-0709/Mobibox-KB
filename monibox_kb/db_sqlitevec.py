"""
db_sqlitevec.py（最终稳健版：用 float32 BLOB 写入 vec0）

解决的问题：
1) Windows/Conda sqlite3 默认禁用扩展加载 -> enable_load_extension(True)
2) 你的 sqlite_vec 包没有 serialize() -> 我们不用 serialize
3) 避免依赖 vec_f32() 是否存在/命令行引号问题 -> 直接写 float32 BLOB
4) rowid 强制 int，避免 numpy 类型导致 sqlite3 绑定失败
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
    """
    把 Python list[float] 转成 float32 little-endian 连续内存的 BLOB。
    vec0 的 embedding 列底层就是吃这种格式（与 vec_f32() 的输出一致）。
    """
    # 强制转 float，避免 numpy.float32/Decimal 等导致 pack 出错
    floats = [float(x) for x in vec]
    blob = struct.pack("<%sf" % len(floats), *floats)  # little-endian float32
    return sqlite3.Binary(blob)


class RagDB:
    def __init__(self, db_path: str, schema_path: Optional[Path] = None):
        self.db_path = db_path
        self.schema_path = schema_path or (SQL_DIR / "schema.sql")

    def connect(self) -> sqlite3.Connection:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # 允许加载扩展（否则 not authorized）
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
                # 1) 写 chunks 元数据
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

                # 2) 写 vec_chunks 向量（BLOB）
                rowid = int(cur.lastrowid)
                blob = vec_to_f32_blob(v)

                cur.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob)
                )

            conn.commit()