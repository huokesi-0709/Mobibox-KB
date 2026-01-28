from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import Counter

import sqlite_vec

from monibox_kb.embedding import embed_texts
from monibox_kb.scoring.rerank import RerankPolicy, final_distance
from monibox_kb.routing.router import AutoRouter


def vec_to_f32_blob(vec: List[float]) -> sqlite3.Binary:
    floats = [float(x) for x in vec]
    blob = struct.pack("<%sf" % len(floats), *floats)
    return sqlite3.Binary(blob)


@dataclass
class SearchResult:
    chunk_id: str
    display_id: Optional[str]
    group_id: Optional[str]
    text: str
    dimension: str
    risk: str
    source_id: str
    status: str
    quality_score: float
    distance: float
    final_distance: float


class RagEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.policy = RerankPolicy.load_default()
        self.router = AutoRouter()

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def search(self,
               query: str,
               topk: int = 5,
               pool_mult: int = 8,
               dimension: Optional[str] = None,
               tags: Optional[List[str]] = None,
               status_exclude: str = "停用",
               max_per_group: int = 1) -> List[SearchResult]:

        qvec = embed_texts([query])[0]
        qblob = vec_to_f32_blob(qvec)

        where = [f"c.status <> :ex_status"]
        params: Dict[str, Any] = {"ex_status": status_exclude}

        if dimension:
            where.append("c.dimension = :dimension")
            params["dimension"] = dimension

        if tags:
            ors = []
            for i, t in enumerate(tags):
                k = f"t{i}"
                params[k] = f"%|{t}|%"
                ors.append(f"c.tags_flat LIKE :{k}")
            where.append("(" + " OR ".join(ors) + ")")

        where_sql = " WHERE " + " AND ".join(where)

        k_pool = min(max(topk, topk * pool_mult), 300)

        sql = f"""
        WITH knn AS (
          SELECT rowid, distance
          FROM vec_chunks
          WHERE embedding MATCH :qvec
            AND k = :kpool
        )
        SELECT
          c.chunk_id, c.display_id, c.group_id,
          c.text, c.dimension, c.risk, c.source_id, c.status, c.quality_score,
          knn.distance
        FROM knn
        JOIN chunks c ON c.id = knn.rowid
        {where_sql}
        ORDER BY knn.distance
        LIMIT :kpool;
        """

        params["qvec"] = qblob
        params["kpool"] = int(k_pool)

        conn = self._open_db()
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        scored = []
        for r in rows:
            d = float(r["distance"])
            q = float(r["quality_score"])
            st = r["status"]
            d_final = final_distance(d, q, st, self.policy)
            scored.append((d_final, r))
        scored.sort(key=lambda x: x[0])

        # diversify by group_id
        picked = []
        group_cnt = Counter()
        for d_final, r in scored:
            gid = r["group_id"] or ""
            if gid and group_cnt[gid] >= max_per_group:
                continue
            picked.append((d_final, r))
            if gid:
                group_cnt[gid] += 1
            if len(picked) >= topk:
                break

        if len(picked) < topk:
            exists = set((r["chunk_id"] for _, r in picked))
            for d_final, r in scored:
                if r["chunk_id"] in exists:
                    continue
                picked.append((d_final, r))
                if len(picked) >= topk:
                    break

        out: List[SearchResult] = []
        for d_final, r in picked:
            out.append(SearchResult(
                chunk_id=r["chunk_id"],
                display_id=r["display_id"],
                group_id=r["group_id"],
                text=r["text"],
                dimension=r["dimension"],
                risk=r["risk"],
                source_id=r["source_id"],
                status=r["status"],
                quality_score=float(r["quality_score"]),
                distance=float(r["distance"]),
                final_distance=float(d_final),
            ))
        return out

    def auto_search(self, query: str, topk: int = 5, auto_top_tags: int = 2) -> List[SearchResult]:
        rr = self.router.route(query, top_tags=auto_top_tags)
        # 跨维度：不锁 dimension，只用 tags
        dim = None if rr.cross_dimension else rr.dimension
        return self.search(query, topk=topk, dimension=dim, tags=rr.tags)