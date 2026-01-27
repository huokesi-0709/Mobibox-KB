from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from monibox_kb.paths import GENERATED_DIR


@dataclass
class RouteResult:
    dimension: str
    tags: List[str]                 # 推荐用于过滤的 tag_id（按重要性排序）
    evidence: Dict[str, List[str]]  # tag_id -> 命中的召回词列表（可解释）


class AutoRouter:
    """
    轻量路由器：
    - 读取 normalized taxonomy
    - 用“建议召回词”做 substring 匹配
    - 输出推荐维度 + 标签
    """
    def __init__(self, normalized_path=None):
        self.normalized_path = normalized_path or (GENERATED_DIR / "02_meta_candidates.normalized.json")
        self.tag_records = []  # [{tag_id, dim, recall_terms}]
        self._load()

    def _load(self):
        if not self.normalized_path.exists():
            raise FileNotFoundError(f"缺少 {self.normalized_path}，请先运行 taxonomy + normalize")

        obj = json.loads(self.normalized_path.read_text(encoding="utf-8"))
        items = obj.get("标签体系", [])
        if not isinstance(items, list):
            raise ValueError("normalized taxonomy 结构不对：标签体系不是数组")

        recs = []
        for it in items:
            if not isinstance(it, dict):
                continue
            tid = str(it.get("标签ID", "")).strip()
            dim = str(it.get("所属维度", "")).strip()
            recall = it.get("建议召回词", [])
            if not tid or not dim:
                continue
            if not isinstance(recall, list):
                recall = []
            recall = [str(x).strip() for x in recall if str(x).strip()]
            recs.append({"tag_id": tid, "dimension": dim, "recall": recall})

        self.tag_records = recs

    def route(self, query: str, top_tags: int = 1) -> RouteResult:
        q = (query or "").strip()
        if not q:
            return RouteResult(dimension="动态心理认知状态", tags=[], evidence={})

        # 1) 计算每个tag的匹配分数
        tag_score: Dict[str, float] = {}
        tag_hits: Dict[str, List[str]] = {}
        tag_dim: Dict[str, str] = {}

        for rec in self.tag_records:
            tid = rec["tag_id"]
            dim = rec["dimension"]
            tag_dim[tid] = dim

            hits = []
            score = 0.0
            for term in rec["recall"]:
                # substring match：足够轻量，端侧也能跑
                if term and term in q:
                    hits.append(term)
                    # 简单打分：命中一次 + 词越长越重要（避免“黑/痛”这种太泛）
                    score += 1.0 + min(len(term), 8) * 0.08

            if score > 0:
                tag_score[tid] = score
                tag_hits[tid] = hits

        # 2) 如果没有命中任何召回词：默认心理维度，不加 tag 过滤
        if not tag_score:
            return RouteResult(dimension="动态心理认知状态", tags=[], evidence={})

        # 3) 选出 top_tags 个 tag
        sorted_tags = sorted(tag_score.items(), key=lambda x: x[1], reverse=True)
        picked = [tid for tid, _ in sorted_tags[:max(1, int(top_tags))]]

        # 4) 推断维度：按命中 tag 的得分累加
        dim_score: Dict[str, float] = {}
        for tid, sc in tag_score.items():
            dim = tag_dim.get(tid, "动态心理认知状态")
            dim_score[dim] = dim_score.get(dim, 0.0) + sc

        # 选分数最高的维度
        best_dim = sorted(dim_score.items(), key=lambda x: x[1], reverse=True)[0][0]

        evidence = {tid: tag_hits.get(tid, []) for tid in picked}
        return RouteResult(dimension=best_dim, tags=picked, evidence=evidence)