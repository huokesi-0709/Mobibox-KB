"""
normalize_taxonomy.py（正确版：纯本地规范化，不调用DeepSeek）

输入：
- knowledge_src/generated/02_meta_candidates.json

输出：
- knowledge_src/generated/02_meta_candidates.normalized.json

作用：
- 统一类别名称（场景/环境/医学/心理/特殊群体/干预动作/设备交互/语言方言/次生灾害）
- 统一所属维度（必须落在五维度内）
- 统一标签ID命名风格（lower_snake_case）
- 强制加前缀（scn_/env_/med_/psy_/spc_/act_/dev_/lang_/sec_）
- 建议召回词/适用人群建议统一为数组
- 去重（按标签ID）
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from monibox_kb.paths import GENERATED_DIR as GEN, KNOWLEDGE_SRC as SRC

CANON_CATEGORIES = [
    "场景", "环境", "医学", "心理", "特殊群体", "干预动作", "设备交互", "语言方言", "次生灾害"
]

CATEGORY_PREFIX = {
    "场景": "scn_",
    "环境": "env_",
    "医学": "med_",
    "心理": "psy_",
    "特殊群体": "spc_",
    "干预动作": "act_",
    "设备交互": "dev_",
    "语言方言": "lang_",
    "次生灾害": "sec_"
}

def load_dims() -> List[str]:
    meta = json.loads((SRC / "00_meta.json").read_text(encoding="utf-8"))
    return meta["枚举"]["五维度"]

def slugify(s: str) -> str:
    s = (s or "").strip().lower().replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        parts = re.split(r"[，,、\s]+", s)
        return [p.strip() for p in parts if p.strip()]
    return [str(v).strip()]

def normalize_category(cat: str, dim: str) -> str:
    c = (cat or "").strip()
    d = (dim or "").strip()

    if c in CANON_CATEGORIES:
        return c

    # 类别被写成“维度名”的情况
    if "次生灾害" in c or "余震" in c or "火灾" in c or "洪水" in c:
        return "次生灾害"

    # 关键词归一
    if "设备" in c or "交互" in c:
        return "设备交互"
    if "方言" in c or "语言" in c:
        return "语言方言"
    if "干预" in c or "安抚" in c or "呼吸" in c or "锚定" in c:
        return "干预动作"
    if "心理" in c or "恐慌" in c or "绝望" in c or "愤怒" in c:
        return "心理"
    if "医学" in c or "病理" in c or "出血" in c or "气道" in c:
        return "医学"
    if "环境" in c or "粉尘" in c or "缺氧" in c or "瓦斯" in c:
        return "环境"
    if "儿童" in c or "老年" in c or "自闭" in c or "特殊" in c:
        return "特殊群体"
    if "场景" in c or "受困" in c or "废墟" in c or "黑暗" in c:
        return "场景"

    # 维度兜底推断
    if "次生灾害" in d:
        return "次生灾害"

    return "场景"

def normalize_dimension(dim: str, dims_allowed: List[str], category: str) -> str:
    d = (dim or "").strip()
    if d in dims_allowed:
        return d

    if category in ("场景", "环境"):
        return "微环境物理特征"
    if category == "次生灾害":
        return "极端环境适应与次生灾害"
    if category == "医学":
        return "核心生理病理学"
    if category in ("心理", "干预动作"):
        return "动态心理认知状态"
    if category == "特殊群体":
        return "特殊群体干预策略"

    return dims_allowed[0]

def ensure_prefix(tag_id: str, category: str) -> str:
    tid = slugify(tag_id)
    prefix = CATEGORY_PREFIX.get(category, "")
    if prefix and not tid.startswith(prefix):
        tid = prefix + tid
    return tid

def main():
    dims_allowed = load_dims()

    in_path = GEN / "02_meta_candidates.json"
    if not in_path.exists():
        raise FileNotFoundError(f"找不到 {in_path}，请先运行 generate_taxonomy.py 生成候选标签")

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    items = raw.get("标签体系", [])
    if not isinstance(items, list):
        raise ValueError("02_meta_candidates.json 结构不对：标签体系 不是数组")

    normalized = []
    seen = set()

    for it in items:
        if not isinstance(it, dict):
            continue

        name = str(it.get("名称", "")).strip()
        cat_raw = str(it.get("类别", "")).strip()
        dim_raw = str(it.get("所属维度", "")).strip()

        cat = normalize_category(cat_raw, dim_raw)
        dim = normalize_dimension(dim_raw, dims_allowed, cat)

        tag_id = ensure_prefix(str(it.get("标签ID", "")), cat)
        if not tag_id or tag_id in seen:
            continue
        seen.add(tag_id)

        recall = to_list(it.get("建议召回词"))[:12]
        pops = to_list(it.get("适用人群建议"))[:6]
        risk = str(it.get("风险等级建议", "中")).strip()
        if risk not in ("低", "中", "高", "致命"):
            risk = "中"

        normalized.append({
            "标签ID": tag_id,
            "名称": name or tag_id,
            "类别": cat,
            "所属维度": dim,
            "建议召回词": recall,
            "适用人群建议": pops,
            "风险等级建议": risk
        })

    # 排序：类别 -> 标签ID
    def cat_idx(c: str) -> int:
        return CANON_CATEGORIES.index(c) if c in CANON_CATEGORIES else 999

    normalized.sort(key=lambda x: (cat_idx(x["类别"]), x["标签ID"]))

    out_path = GEN / "02_meta_candidates.normalized.json"
    out_path.write_text(json.dumps({"标签体系": normalized}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE. normalized saved:", out_path, "count=", len(normalized))

if __name__ == "__main__":
    main()