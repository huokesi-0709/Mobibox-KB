from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from monibox_kb.paths import KNOWLEDGE_SRC

class ProtocolEngine:
    """
    协议优先引擎（增强版）：
    - 支持 all_of / any_of / none_of 组合条件
    - 条件类型：events / text_contains_any / tags_any / tags_all
    """
    def __init__(self, protocols_path: Optional[Path] = None):
        self.protocols_path = protocols_path or (KNOWLEDGE_SRC / "protocols.json")
        obj = json.loads(self.protocols_path.read_text(encoding="utf-8"))
        ps = obj.get("protocols", [])
        self.protocols = sorted(ps, key=lambda x: x.get("priority", 0), reverse=True)

    def match(self, text: str, routed_tags: List[str], events: List[str]) -> Optional[Dict[str, Any]]:
        text = text or ""
        routed_tags = routed_tags or []
        events = events or []

        for p in self.protocols:
            trig = p.get("trigger", {})
            if self._eval_trigger(trig, text, routed_tags, events):
                return p
        return None

    def _eval_trigger(self, trig: Dict[str, Any], text: str, tags: List[str], events: List[str]) -> bool:
        # 默认：any_of 为空则视为 False（避免误触发）
        any_of = trig.get("any_of", [])
        all_of = trig.get("all_of", [])
        none_of = trig.get("none_of", [])

        if all_of and not self._all(all_of, text, tags, events):
            return False
        if any_of and not self._any(any_of, text, tags, events):
            return False
        if none_of and self._any(none_of, text, tags, events):
            return False

        # 如果三者都为空，认为不触发
        if not any_of and not all_of and not none_of:
            return False
        return True

    def _any(self, conds: List[Dict[str, Any]], text: str, tags: List[str], events: List[str]) -> bool:
        for c in conds:
            if self._match_one(c, text, tags, events):
                return True
        return False

    def _all(self, conds: List[Dict[str, Any]], text: str, tags: List[str], events: List[str]) -> bool:
        for c in conds:
            if not self._match_one(c, text, tags, events):
                return False
        return True

    def _match_one(self, cond: Dict[str, Any], text: str, tags: List[str], events: List[str]) -> bool:
        # event
        if "event" in cond:
            return cond["event"] in events

        # text contains any
        if "text_contains_any" in cond:
            for w in cond["text_contains_any"]:
                if w and w in text:
                    return True
            return False

        # tags any
        if "tags_any" in cond:
            return any(t in tags for t in cond["tags_any"])

        # tags all
        if "tags_all" in cond:
            return all(t in tags for t in cond["tags_all"])

        return False