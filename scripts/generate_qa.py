"""
generate_qa.py（分页补齐版本）

解决问题：
- 模型经常“不按要求输出足量条数”
- 一次生成太多也容易截断
=> 分页生成，每批 10~20 条，直到补齐 target_total

输出：
- knowledge_src/generated/11_qa_synth.json
调试输出：
- knowledge_src/generated/_debug_qa_raw_batchXX.txt
"""

import json
import uuid
import hashlib

from monibox_kb.deepseek_client import DeepSeekClient
from monibox_kb.utils_json import extract_json
from monibox_kb.paths import KNOWLEDGE_SRC as SRC, GENERATED_DIR as OUT

OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是MoniBox问答数据生成器。只能输出JSON数组，不要解释，不要Markdown，不要代码块。"

REQUIRED_KEYS = ["维度", "风险等级", "适用人群", "标签", "召回词", "问题", "回答", "安全提示"]


def fp_qa(q: str, a: str) -> str:
    s = (q.strip() + "\n" + a.strip()).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def build_prompt(dims, risks, pops, tag_preview, n: int) -> str:
    return f"""
生成 {n} 条问答数据（JSON数组），字段必须固定为：
qa_id, 维度, 子主题, 风险等级, 适用人群, 标签, 召回词, 问题, 回答, 安全提示

硬性约束：
1) 必须只输出严格JSON数组（不要任何额外文字、不要```围栏）。
2) 维度只能从：{dims}
3) 风险等级只能从：{risks}
4) 适用人群只能从：{pops}
5) 标签必须是 lower_snake_case 标签ID数组，只能从此集合中选：{tag_preview}
6) 召回词必须是数组，最多 10 个
7) 回答必须短句、可TTS播报、非诊断性、不提供剂量、不承诺获救时间。
8) 医学相关允许出现，但只能给自我保护/节省体力/合理求救/等待救援建议，禁止高风险医疗操作指导。
9) 问题要多样，覆盖：恐慌、愤怒、悲伤、绝望、儿童、老年、呼吸不适、余震等。

开始输出JSON数组：
""".strip()


def main():
    meta = json.loads((SRC / "00_meta.json").read_text(encoding="utf-8"))

    # 优先使用 normalized taxonomy（更整齐）
    cand_norm = OUT / "02_meta_candidates.normalized.json"
    cand_raw = OUT / "02_meta_candidates.json"
    cand_path = cand_norm if cand_norm.exists() else cand_raw
    candidates = json.loads(cand_path.read_text(encoding="utf-8"))

    dims = meta["枚举"]["五维度"]
    risks = meta["枚举"]["风险等级"]
    pops = meta["枚举"]["人群"]

    tag_ids = [t["标签ID"] for t in meta.get("标签体系", [])] + [t["标签ID"] for t in candidates.get("标签体系", [])]
    tag_ids = sorted(list(set(tag_ids)))
    tag_preview = tag_ids[:280]  # 防止 prompt 过长

    target_total = 40     # 你要多少条 QA
    batch_size = 12       # 每批生成多少（10~20比较稳）
    max_batches = 20      # 最多批次数，防止无限循环

    cli = DeepSeekClient()

    out_list = []
    seen_fp = set()

    batch_no = 0
    while len(out_list) < target_total and batch_no < max_batches:
        batch_no += 1
        need = min(batch_size, target_total - len(out_list))

        print(f"[Batch {batch_no}] 目标再生成 {need} 条（当前已有 {len(out_list)}/{target_total}）...")
        raw = cli.chat(SYSTEM, build_prompt(dims, risks, pops, tag_preview, need), temperature=0.7)

        raw_path = OUT / f"_debug_qa_raw_batch{batch_no:02d}.txt"
        raw_path.write_text(raw or "", encoding="utf-8")
        print(f"    debug raw saved: {raw_path.name} (len={len(raw or '')})")

        data = extract_json(raw)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list):
            print("    [WARN] 输出不是数组，跳过本批")
            continue

        added = 0
        for it in data:
            if not isinstance(it, dict):
                continue
            if any(k not in it for k in REQUIRED_KEYS):
                continue

            q, a = it.get("问题", ""), it.get("回答", "")
            key = fp_qa(q, a)
            if key in seen_fp:
                continue
            seen_fp.add(key)

            it["qa_id"] = it.get("qa_id") or ("qa_" + uuid.uuid4().hex[:12])
            it["来源ID"] = "src_synth_deepseek_v1"
            it["状态"] = "候选"
            it["人工评分"] = 0

            out_list.append(it)
            added += 1
            if len(out_list) >= target_total:
                break

        print(f"    本批新增 {added}，累计 {len(out_list)}/{target_total}")

    out_path = OUT / "11_qa_synth.json"
    out_path.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE. saved:", out_path, "count=", len(out_list))


if __name__ == "__main__":
    main()