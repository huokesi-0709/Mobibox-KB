"""
qa_to_chunks.py（最终稳健版）

目的：
- 将 QA 的“回答/答案/回复/解答”切成 60 字原子片段（TTS友好）
- 自动添加来源/状态/评分/指纹
- 重点修复：片段ID重复问题
  原因：QA 里的 qa_id 可能是 1、2、3... 且会重复
  解决：为每条 QA 生成稳定且唯一的 片段组ID(qa_gid)，并在片段ID中加入指纹短码

输出：
- knowledge_src/generated/12_chunks_synth.json
如有坏条目（缺回答等）：
- knowledge_src/generated/11_qa_synth.bad.json
"""

import json
import re
import hashlib
from typing import Any, Dict, List

from monibox_kb.config import settings
from monibox_kb.chunking import split_by_max_chars
from monibox_kb.text_clean import clean_text
from monibox_kb.dedup import sha256_fp
from monibox_kb.paths import GENERATED_DIR as GEN

OUT_PATH = GEN / "12_chunks_synth.json"
BAD_QA_PATH = GEN / "11_qa_synth.bad.json"


# ========= 可读ID辅助：维度/来源/标签短名 =========

def dim_short(dim: str) -> str:
    m = {
        "核心生理病理学": "med",
        "动态心理认知状态": "psy",
        "微环境物理特征": "env",
        "极端环境适应与次生灾害": "sec",
        "特殊群体干预策略": "spc",
    }
    return m.get((dim or "").strip(), "misc")


def source_short(source_id: str) -> str:
    s = (source_id or "").lower()
    if "deepseek" in s or "synth" in s:
        return "synth"
    if "expert" in s:
        return "expert"
    if "guideline" in s or "who" in s:
        return "guide"
    return "src"


def tag_short(tag_id: str) -> str:
    x = (tag_id or "").strip().lower().replace("-", "_")
    parts = [p for p in x.split("_") if p]
    return parts[-1] if parts else "tag"


# ========= key 归一化（避免 “回答：” 这种） =========

def normalize_key(k: str) -> str:
    if not isinstance(k, str):
        return str(k)
    k = k.strip()
    k = re.sub(r"[：:]\s*$", "", k)  # 去尾部冒号
    k = re.sub(r"\s+", "", k)       # 去全部空白
    return k


def normalize_item_keys(it: Any) -> Dict[str, Any]:
    if not isinstance(it, dict):
        return {}
    out = {}
    for k, v in it.items():
        out[normalize_key(k)] = v
    return out


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


def pick_answer(it: Dict[str, Any]) -> str:
    for key in ("回答", "答案", "回复", "解答"):
        v = it.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


# ========= 关键：为每条QA生成稳定且唯一的 片段组ID =========

def make_qa_gid(it: Dict[str, Any], idx: int) -> str:
    """
    生成片段组ID（qa_gid），保证唯一且尽量稳定：
    - 不相信 it['qa_id']（可能是 1,2,3 且重复）
    - 用 idx + (问题+回答) 的 hash 生成
    """
    q = it.get("问题", "")
    a = pick_answer(it)
    base = f"{idx}||{str(q)}||{str(a)}".encode("utf-8", errors="ignore")
    h = hashlib.sha1(base).hexdigest()[:10]  # 10位足够避免冲突
    return f"qa_{idx:05d}_{h}"


def main():
    qa_path = GEN / "11_qa_synth.json"
    if not qa_path.exists():
        raise FileNotFoundError("缺少 11_qa_synth.json，请先运行 scripts/generate_qa.py")

    raw = json.loads(qa_path.read_text(encoding="utf-8"))

    # 兼容 {"data":[...]}
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        qa_list = raw["data"]
    else:
        qa_list = raw

    if not isinstance(qa_list, list):
        raise ValueError("11_qa_synth.json 结构不对：根不是数组（或 dict.data 数组）")

    chunks = []
    bad_items = []

    for idx, it0 in enumerate(qa_list):
        it = normalize_item_keys(it0)

        answer = pick_answer(it)
        if not answer:
            bad_items.append({
                "__index": idx,
                "__reason": "missing_answer_field",
                "__keys": list(it.keys()),
                "__item": it0
            })
            continue

        answer = clean_text(answer)
        atoms = split_by_max_chars(answer, settings.chunk_max_chars, settings.chunk_min_chars)
        if not atoms:
            bad_items.append({
                "__index": idx,
                "__reason": "empty_answer_after_clean_or_split",
                "__keys": list(it.keys()),
                "__item": it0
            })
            continue

        # 召回词 + 问题（提高命中）
        recall_terms = set(to_list(it.get("召回词")))
        q = it.get("问题")
        if isinstance(q, str) and q.strip():
            recall_terms.add(q.strip())

        # 不再使用 it['qa_id'] 来生成片段ID，避免重复
        qa_gid = make_qa_gid(it, idx)

        # 元字段
        dim = it.get("维度", "动态心理认知状态")
        risk = it.get("风险等级", "中")
        pops = to_list(it.get("适用人群"))
        tags = to_list(it.get("标签"))
        primary_tag = tags[0] if tags else "tag_unknown"

        src_id = it.get("来源ID", "src_synth_deepseek_v1")
        status = it.get("状态", "候选")
        score = it.get("人工评分", 0)
        safety_tip = it.get("安全提示", it.get("__安全提示", ""))

        for i, atom in enumerate(atoms):
            fp = sha256_fp(atom)                 # sha256:xxxx
            fp8 = fp.split(":")[1][:8]           # 指纹短码

            # 机器唯一ID：加入 qa_gid + 指纹短码，彻底避免重复
            chunk_id = f"k_{qa_gid}_{i:02d}_{fp8}"

            # 人类可读ID：方便测试人员点名（不要求唯一，但我们也让它基本唯一）
            # 示例：k_synth_psy_panic_9106c91e_00
            display_id = f"k_{source_short(src_id)}_{dim_short(dim)}_{tag_short(primary_tag)}_{fp8}_{i:02d}"

            chunk = {
                "类型": "知识片段",
                "片段ID": chunk_id,       # 数据库主键（必须唯一）
                "显示ID": display_id,     # 测试/日志用（好认）
                "片段组ID": qa_gid,       # 同一条QA切分后的组ID

                "维度": dim,
                "子主题": it.get("子主题"),
                "风险等级": risk,
                "适用人群": pops,
                "标签": tags,
                "召回词": sorted([x for x in recall_terms if x])[:80],

                "可直接播报": True,
                "播报风格": "冷静清晰",
                "文本": atom,

                "来源ID": src_id,
                "状态": status,
                "人工评分": score,
                "内容指纹": fp,

                "__安全提示": safety_tip
            }
            chunks.append(chunk)

    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUT_PATH, "count=", len(chunks))

    if bad_items:
        BAD_QA_PATH.write_text(json.dumps(bad_items, ensure_ascii=False, indent=2), encoding="utf-8")
        print("WARN: found bad QA items:", len(bad_items), "saved to:", BAD_QA_PATH)
    else:
        print("No bad QA items.")


if __name__ == "__main__":
    main()