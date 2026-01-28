from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from monibox_kb.paths import GENERATED_DIR, KNOWLEDGE_SRC
from monibox_kb.tags.registry import TagRegistry


@dataclass
class RouteResult:
    dimension: str
    tags: List[str]                  # 推荐用于过滤的 tag_id（按重要性排序）
    tag_dims: Dict[str, str]         # tag_id -> 所属维度
    evidence: Dict[str, List[str]]   # tag_id -> 命中的召回词/规则词（可解释）
    cross_dimension: bool            # 是否跨维度（用于 query_demo 解锁维度过滤）


class AutoRouter:
    """
    轻量路由器（端侧可用）：
    - 基于 normalized taxonomy 的“建议召回词”做匹配
    - 支持 router_overrides.json 对特定症状/关键词加权（不用改taxonomy）
    - 输出：tags + tag_dims + cross_dimension
    """
    def __init__(self,
                 normalized_path=None,
                 overrides_path=None):
        self.normalized_path = normalized_path or (GENERATED_DIR / "02_meta_candidates.normalized.json")
        self.overrides_path = overrides_path or (KNOWLEDGE_SRC / "router_overrides.json")

        self.tag_records = []            # [{tag_id, dim, recall[]}]
        self.tag_dim_map: Dict[str, str] = {}
        self.overrides = []              # rules
        self.reg = TagRegistry.load()    # 含 alias + allowed_ids

        self._load_taxonomy()
        self._load_overrides()

    def _load_taxonomy(self):
        if not self.normalized_path.exists():
            raise FileNotFoundError(f"缺少 {self.normalized_path}，请先生成并 normalize taxonomy")

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
            self.tag_dim_map[tid] = dim

        self.tag_records = recs

    def _load_overrides(self):
        if not self.overrides_path.exists():
            self.overrides = []
            return
        obj = json.loads(self.overrides_path.read_text(encoding="utf-8"))
        rules = obj.get("rules", [])
        if isinstance(rules, list):
            self.overrides = rules
        else:
            self.overrides = []

    def _apply_overrides(self, q: str,
                         tag_score: Dict[str, float],
                         tag_hits: Dict[str, List[str]]):
        """
        对匹配到 patterns 的规则，给 boost_tags 加分；
        force_tags 则给一个大分确保进入 top_tags（仍会走 canonicalize）。
        """
        for rule in self.overrides:
            try:
                patterns = rule.get("patterns", [])
                if not isinstance(patterns, list):
                    continue
                hit_terms = [p for p in patterns if isinstance(p, str) and p and (p in q)]
                if not hit_terms:
                    continue

                # boost tags
                boost = rule.get("boost_tags", {})
                if isinstance(boost, dict):
                    for raw_tid, w in boost.items():
                        canon = self.reg.canonicalize(str(raw_tid))
                        if not canon:
                            continue
                        if canon not in self.tag_dim_map:
                            continue
                        tag_score[canon] = tag_score.get(canon, 0.0) + float(w)
                        tag_hits.setdefault(canon, []).extend(hit_terms)

                # force tags
                force = rule.get("force_tags", [])
                if isinstance(force, list):
                    for raw_tid in force:
                        canon = self.reg.canonicalize(str(raw_tid))
                        if not canon:
                            continue
                        if canon not in self.tag_dim_map:
                            continue
                        tag_score[canon] = tag_score.get(canon, 0.0) + 99.0
                        tag_hits.setdefault(canon, []).extend(hit_terms)

            except Exception:
                continue

    def route(self, query: str, top_tags: int = 1) -> RouteResult:
        q = (query or "").strip()
        if not q:
            return RouteResult(
                dimension="动态心理认知状态",
                tags=[],
                tag_dims={},
                evidence={},
                cross_dimension=False
            )

        tag_score: Dict[str, float] = {}
        tag_hits: Dict[str, List[str]] = {}

        # 1) 召回词匹配
        for rec in self.tag_records:
            tid = rec["tag_id"]
            dim = rec["dimension"]

            score = 0.0
            hits = []
            for term in rec["recall"]:
                if term and term in q:
                    hits.append(term)
                    # 命中一次 + 长词加权（避免“黑/痛”过泛）
                    score += 1.0 + min(len(term), 8) * 0.08

            if score > 0:
                tag_score[tid] = tag_score.get(tid, 0.0) + score
                tag_hits.setdefault(tid, []).extend(hits)

        # 2) overrides 加权（关键：不用改 taxonomy）
        self._apply_overrides(q, tag_score, tag_hits)

        # 3) 若完全无命中：默认心理维度，不加 tag 过滤
        if not tag_score:
            return RouteResult(
                dimension="动态心理认知状态",
                tags=[],
                tag_dims={},
                evidence={},
                cross_dimension=False
            )

        # 4) 选 top_tags
        top_tags = max(1, int(top_tags))
        sorted_tags = sorted(tag_score.items(), key=lambda x: x[1], reverse=True)
        picked = [tid for tid, _ in sorted_tags[:top_tags]]

        # 5) 推断维度：按命中得分累加
        dim_score: Dict[str, float] = {}
        for tid, sc in tag_score.items():
            dim = self.tag_dim_map.get(tid, "动态心理认知状态")
            dim_score[dim] = dim_score.get(dim, 0.0) + sc
        best_dim = sorted(dim_score.items(), key=lambda x: x[1], reverse=True)[0][0]

        # 6) cross-dimension 判断
        picked_dims = {self.tag_dim_map.get(tid, "") for tid in picked if tid in self.tag_dim_map}
        cross_dim = len({d for d in picked_dims if d}) >= 2

        evidence = {tid: tag_hits.get(tid, []) for tid in picked}
        tag_dims = {tid: self.tag_dim_map.get(tid, "") for tid in picked}

        return RouteResult(
            dimension=best_dim,
            tags=picked,
            tag_dims=tag_dims,
            evidence=evidence,
            cross_dimension=cross_dim
        )
