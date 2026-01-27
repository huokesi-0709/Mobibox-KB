from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from monibox_kb.paths import PROJECT_ROOT


@dataclass
class RerankPolicy:
    w_quality: float = 0.015   # 中等强度
    w_enabled: float = 0.005

    @staticmethod
    def load_default() -> "RerankPolicy":
        """
        从 scoring_system/policy.json 读取策略。
        读取失败则使用默认值（中等强度）。
        """
        p = PROJECT_ROOT / "scoring_system" / "policy.json"
        if not p.exists():
            return RerankPolicy()

        obj = json.loads(p.read_text(encoding="utf-8"))
        w = obj.get("weights", {})
        return RerankPolicy(
            w_quality=float(w.get("w_quality", 0.015)),
            w_enabled=float(w.get("w_enabled", 0.005)),
        )


def final_distance(distance: float, quality_score: float, status: str, policy: RerankPolicy) -> float:
    """
    distance 越小越好
    通过减小 distance 来提升排序
    """
    q = max(0.0, min(5.0, float(quality_score)))
    enabled = 1.0 if (status == "启用") else 0.0
    return float(distance) - policy.w_quality * (q / 5.0) - policy.w_enabled * enabled