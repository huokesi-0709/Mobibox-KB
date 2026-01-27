from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

from monibox_kb.paths import GENERATED_DIR, KNOWLEDGE_SRC


def slugify(s: str) -> str:
    s = (s or "").strip().lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass
class TagRegistry:
    """
    标签注册表（标签收敛核心）
    - allowed_ids: 允许使用的标准 tag_id 集合（来自 meta + normalized taxonomy）
    - name_to_id: 中文名称 -> 标准 tag_id（轻度纠错）
    - alias_map : 旧tag -> [标准tag候选列表]（收敛遗留标签）
    """
    allowed_ids: Set[str]
    name_to_id: Dict[str, str]
    alias_map: Dict[str, List[str]]

    @staticmethod
    def load(
        meta_path: Optional[Path] = None,
        normalized_path: Optional[Path] = None,
        alias_path: Optional[Path] = None
    ) -> "TagRegistry":
        meta_path = meta_path or (KNOWLEDGE_SRC / "00_meta.json")
        normalized_path = normalized_path or (GENERATED_DIR / "02_meta_candidates.normalized.json")
        alias_path = alias_path or (KNOWLEDGE_SRC / "tag_alias.json")

        allowed: Set[str] = set()
        name_to_id: Dict[str, str] = {}
        alias_map: Dict[str, List[str]] = {}

        # 1) base meta tags
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for t in meta.get("标签体系", []):
            tid = slugify(t.get("标签ID", ""))
            if tid:
                allowed.add(tid)
                if t.get("名称"):
                    name_to_id[str(t["名称"]).strip()] = tid

        # 2) normalized candidates
        if normalized_path.exists():
            obj = json.loads(normalized_path.read_text(encoding="utf-8"))
            for t in obj.get("标签体系", []):
                tid = slugify(t.get("标签ID", ""))
                if tid:
                    allowed.add(tid)
                    if t.get("名称"):
                        name_to_id[str(t["名称"]).strip()] = tid

        # 3) alias map (optional)
        if alias_path.exists():
            aobj = json.loads(alias_path.read_text(encoding="utf-8"))
            aliases = aobj.get("aliases", {})
            if isinstance(aliases, dict):
                for k, v in aliases.items():
                    key = slugify(str(k))
                    if not key:
                        continue

                    targets: List[str] = []
                    if isinstance(v, list):
                        targets = [slugify(str(x)) for x in v if slugify(str(x))]
                    elif isinstance(v, str):
                        tv = slugify(v)
                        targets = [tv] if tv else []
                    else:
                        # 其它类型忽略
                        targets = []

                    if targets:
                        alias_map[key] = targets

        return TagRegistry(allowed_ids=allowed, name_to_id=name_to_id, alias_map=alias_map)

    def _resolve_alias(self, tag_id: str) -> Optional[str]:
        """
        给一个 tag_id（已 slugify）尝试走 alias。
        返回第一个存在于 allowed_ids 的目标 tag。
        """
        targets = self.alias_map.get(tag_id)
        if not targets:
            return None
        for t in targets:
            if t in self.allowed_ids:
                return t
        return None

    def canonicalize(self, tag: str) -> Optional[str]:
        """
        将输入 tag 归一为 canonical tag_id（alias 优先）：
        1) 中文名称映射（name_to_id）
        2) slugify
        3) alias 映射（即使旧tag在 allowed_ids，也优先映射）
        4) 若仍未映射且 tid 本身在 allowed_ids，则返回 tid
        5) 否则 None
        """
        if not tag:
            return None

        raw = str(tag).strip()
        if not raw:
            return None

        # 1) 中文名称 -> id
        if raw in self.name_to_id:
            return self.name_to_id[raw]

        tid = slugify(raw)

        # 2) alias 优先（关键修复点）
        mapped = self._resolve_alias(tid)
        if mapped:
            return mapped

        # 3) 已是标准 tag
        if tid in self.allowed_ids:
            return tid

        return None

    def canonicalize_list(self, tags: List[str]) -> List[str]:
        out = []
        for t in tags or []:
            c = self.canonicalize(t)
            if c and c not in out:
                out.append(c)
        return out