"""
generate_taxonomy.py（分页版本）

为什么要分页：
- 一次生成 60~120 个标签很容易被 API 截断，导致 JSON 缺尾括号
- 分页每次 20 个左右，稳定可解析，然后合并去重

输出：
- knowledge_src/generated/02_meta_candidates.json
调试输出：
- knowledge_src/generated/_debug_taxonomy_raw_batchXX.txt
"""

import json
from monibox_kb.deepseek_client import DeepSeekClient
from monibox_kb.utils_json import extract_json
from monibox_kb.paths import KNOWLEDGE_SRC as SRC, GENERATED_DIR as OUT

OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是MoniBox离线知识库的体系构建器。只能输出JSON，不要解释，不要Markdown，不要代码块。"


def build_prompt(dims, count: int, avoid_ids: list[str]) -> str:
    # 为了避免提示太长，只给模型最近的一部分“不要重复”的ID
    avoid_preview = avoid_ids[-80:]  # 最近80个足够抑制重复
    return f"""
请生成 {count} 个“标签体系”条目，用于地震受困陪伴设备 MoniBox。

硬性要求：
1) 必须只输出严格JSON对象（不要任何额外文字、不要```围栏）。
2) JSON结构固定为：{{"标签体系":[ ... ]}}
3) 五维度只能从：{dims}
4) 标签类别至少覆盖：场景、环境、医学、心理、特殊群体、干预动作、设备交互、语言方言
5) 标签ID必须 lower_snake_case（不要全大写）
6) 每个标签对象字段必须包含：
   标签ID, 名称, 类别, 所属维度, 建议召回词, 适用人群建议, 风险等级建议
7) 建议召回词：数组，最多 8 个
8) 适用人群建议：数组，最多 4 个
9) 风险等级建议只能是：低/中/高/致命
10) 不要编造医学剂量/诊断结论，偏描述与非诊断性支持
11) 不要生成以下标签ID（避免重复）：{avoid_preview}

开始输出JSON：
""".strip()


def main():
    meta_path = SRC / "00_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 {meta_path}")

    print("[1/5] 读取 meta:", meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = meta["枚举"]["五维度"]

    # 你想要的“总数”与“每页数量”在这里控制
    target_total = 120      # 最终希望得到多少候选标签
    batch_size = 20         # 每次生成多少（越小越稳，15~25都行）
    max_batches = 20        # 最大批次数（防止无限循环）

    cli = DeepSeekClient()

    all_items: list[dict] = []
    seen_ids: set[str] = set()

    print(f"[2/5] 开始分页生成：target_total={target_total}, batch_size={batch_size}")

    batch_no = 0
    while len(seen_ids) < target_total and batch_no < max_batches:
        batch_no += 1
        need = min(batch_size, target_total - len(seen_ids))

        print(f"[3/5] Batch {batch_no}: 生成 {need} 个标签（当前已有 {len(seen_ids)}）...")
        prompt = build_prompt(dims, need, list(seen_ids))
        raw = cli.chat(SYSTEM, prompt, temperature=0.6)

        raw_path = OUT / f"_debug_taxonomy_raw_batch{batch_no:02d}.txt"
        raw_path.write_text(raw or "", encoding="utf-8")
        print(f"      已保存原始输出：{raw_path.name} (len={len(raw or '')})")

        data = extract_json(raw)

        if not isinstance(data, dict) or "标签体系" not in data:
            print("      [WARN] 输出不是预期结构，跳过本批次 ")
            continue

        items = data["标签体系"]
        if not isinstance(items, list):
            print("      [WARN] 标签体系不是数组，跳过本批次")
            continue

        # 合并去重（按 标签ID）
        added = 0
        for it in items:
            tag_id = it.get("标签ID")
            if not tag_id or not isinstance(tag_id, str):
                continue
            if tag_id in seen_ids:
                continue
            seen_ids.add(tag_id)
            all_items.append(it)
            added += 1

        print(f"      本批次新增：{added}，累计：{len(seen_ids)}")

    if len(seen_ids) == 0:
        raise RuntimeError("生成失败：没有得到任何可用标签。请查看 generated/_debug_taxonomy_raw_batch*.txt")

    out_obj = {"标签体系": all_items}

    out_path = OUT / "02_meta_candidates.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[5/5] DONE. saved:", out_path, "count=", len(all_items))


if __name__ == "__main__":
    main()