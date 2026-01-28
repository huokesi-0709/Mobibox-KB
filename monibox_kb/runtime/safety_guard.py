from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GuardResult:
    level: str              # "allow" | "rewrite" | "block"
    reasons: List[str]
    safe_text: str          # allow: 原文; rewrite: 改写后的; block: 安全替代文本


class SafetyGuard:
    """
    灾害救援场景安全护栏（规则版 v2）

    v2 目标：
    - 把“用药/剂量暗示”从 block 降级为 rewrite（避免过度拦截）
    - 不再因为“静脉出血”这类描述误触发输液/注射拦截
    - 修正 mg/ml 的误伤：使用更严格的边界匹配
    - “检查口鼻异物”改为 rewrite（保留头偏向一侧的安全动作）
    """

    def __init__(self):
        # 1) 高风险：侵入性/操作性强（block）
        self.block_patterns: List[Tuple[str, str]] = [
            # 明确侵入性处置
            (r"缝合|切开|穿刺|引流|开口", "invasive_procedure"),
            (r"止血带", "tourniquet_instruction"),

            # 明确注射/输液（注意：不再拦截“静脉出血”这类描述）
            (r"静脉(注射|输液|给药|穿刺)", "iv_instruction"),
            (r"注射|输液|点滴|静脉输液|静脉注射", "iv_instruction"),

            # 强诊断/强保证
            (r"你(一定|肯定)是", "diagnosis_assertion"),
            (r"(马上|立刻)就能获救", "guarantee_rescue_time"),
        ]

        # 2) 中风险：可改写（rewrite）
        self.rewrite_patterns: List[Tuple[str, str]] = [
            # 用药/剂量暗示：改写，不直接 block
            (r"按(常规)?剂量使用", "medication_dosage_hint"),
            (r"剂量", "medication_dosage_hint"),
            (r"按说明书", "medication_generic"),
            (r"按医嘱", "medication_generic"),

            # 口鼻异物检查：改写为更安全的“头偏一侧+保持呼吸通畅”
            (r"检查其口鼻是否有明显异物", "airway_foreign_body_check"),
        ]

        # 3) 更严格的剂量单位匹配（只拦截明确 mg/ml 单位；并且避免误伤）
        # - 对英文单位使用“非字母边界”
        self.dosage_unit_patterns: List[Tuple[str, str]] = [
            (r"毫克|毫升|片|粒", "med_dosage_unit"),
            (r"(?<![A-Za-z])mg(?![A-Za-z])", "med_dosage_unit"),
            (r"(?<![A-Za-z])ml(?![A-Za-z])", "med_dosage_unit"),
        ]

        self.block_fallback = (
            "我不能指导你进行用药剂量或高风险处置。"
            "你先尽量保持呼吸顺畅、减少活动、节省体力，并等待专业救援。"
            "如果你愿意，告诉我：你现在呼吸更像‘喘不上气’，还是‘胸口很闷’？"
        )

    def check(self, text: str) -> GuardResult:
        t = (text or "").strip()
        if not t:
            return GuardResult(level="allow", reasons=[], safe_text=t)

        # 1) block
        reasons = []
        for pat, code in self.block_patterns:
            if re.search(pat, t):
                reasons.append(code)
        if reasons:
            return GuardResult(level="block", reasons=reasons, safe_text=self.block_fallback)

        # 2) dosage unit：只有当文本同时出现“药/用药/服用/喷雾/吸入”等语境才触发 rewrite/block
        # 避免“少量水”这种误伤
        dosage_unit_hit = False
        for pat, code in self.dosage_unit_patterns:
            if re.search(pat, t, flags=re.IGNORECASE):
                dosage_unit_hit = True
                reasons.append(code)

        if dosage_unit_hit:
            # 如果在用药语境中，改写
            if re.search(r"药|服用|吃药|用药|喷雾|吸入", t):
                safe = "我不能提供药物剂量建议。请优先保持呼吸顺畅、减少活动、等待专业救援。"
                return GuardResult(level="rewrite", reasons=reasons, safe_text=safe)
            # 否则忽略（allow）
            reasons = []  # 清空原因
            # 继续走 rewrite_patterns/allow

        # 3) rewrite
        rw = []
        for pat, code in self.rewrite_patterns:
            if re.search(pat, t):
                rw.append(code)

        if rw:
            safe = t

            # 用药/剂量统一改写
            safe = re.sub(r"如果身边有药物，请按常规剂量使用。", "如果你有医生长期让你随身携带的急救药物，请按你最熟悉且安全的方式使用。", safe)
            safe = re.sub(r"按说明书", "按你最熟悉且安全的方式", safe)
            safe = re.sub(r"按医嘱", "按你最熟悉且安全的方式", safe)

            # 口鼻异物检查改写：删掉检查异物，保留安全的头偏一侧提示
            safe = re.sub(r"情况危急。请立即检查其口鼻是否有明显异物，并小心将其头部偏向一侧，保持气道尽可能通畅。",
                          "情况紧急。请尽量让对方头部偏向一侧，保持呼吸尽可能通畅。避免进行可能导致误吸或误伤的操作，等待专业救援。", safe)

            return GuardResult(level="rewrite", reasons=rw, safe_text=safe)

        return GuardResult(level="allow", reasons=[], safe_text=t)