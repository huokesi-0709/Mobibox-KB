"""
generate_taxonomy.py（分页生成 + 自动规范化）

你遇到的问题：
- DeepSeek 一次生成太多会截断 -> 我们分页
- 模型有时不按 count 输出（你要1条它吐6条） -> 我们允许它吐，但最终会去重并截断到 target_total
- 生成结果“杂乱” -> 生成后自动运行 normalize_taxonomy.py 输出 normalized 文件
"""

import json
from monibox_kb.deepseek_client import DeepSeekClient
from monibox_kb.utils_json import extract_json
from monibox_kb.paths import KNOWLEDGE_SRC as SRC, GENERATED_DIR as OUT

OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是MoniBox离线知识库的体系构建器。只能输出JSON，不要解释，不要Markdown，不要代码块。"


def build_prompt(dims, count: int, avoid_ids: list[str]) -> str:
    avoid_preview = avoid_ids[-80:]
    return f"""
请生成 {count} 个“标签体系”条目，用于地震受困陪伴设备 MoniBox。

硬性要求：
1) 必须只输出严格JSON对象（不要任何额外文字、不要```围栏）。
2) JSON结构固定为：{{"标签体系":[ ... ]}}
3) 五维度只能从：{dims}
4) 标签类别至少覆盖：场景、环境、医学、心理、特殊群体、干预动作、设备交互、语言方言、次生灾害
5) 标签ID必须 lower_snake_case（不要全大写）
6) 每个标签对象字段必须包含：
   标签ID, 名称, 类别, 所属维度, 建议召回词, 适用人群建议, 风险等级建议
7) 建议召回词：数组，最多 8 个
8) 适用人群建议：数组，最多 4 个
9) 风险等级建议只能是：低/中/高/致命
10) 不要编造医学剂量/诊断结论
11) 不要生成以下标签ID（避免重复）：{avoid_preview}

开始输出JSON：
""".strip()


def main():
    meta = json.loads((SRC / "00_meta.json").read_text(encoding="utf-8"))
    dims = meta["枚举"]["五维度"]

    target_total = 120
    batch_size = 20
    max_batches = 20

    cli = DeepSeekClient()

    all_items = []
    seen_ids = set()

    print(f"[1/4] 分页生成 taxonomy：target_total={target_total}, batch_size={batch_size}")

    batch_no = 0
    while len(seen_ids) < target_total and batch_no < max_batches:
        batch_no += 1
        need = min(batch_size, target_total - len(seen_ids))

        print(f"[2/4] Batch {batch_no}: 期望生成 {need} 个（已有 {len(seen_ids)}）...")
        raw = cli.chat(SYSTEM, build_prompt(dims, need, list(seen_ids)), temperature=0.6)

        raw_path = OUT / f"_debug_taxonomy_raw_batch{batch_no:02d}.txt"
        raw_path.write_text(raw or "", encoding="utf-8")
        print(f"      debug raw saved: {raw_path.name} (len={len(raw or '')})")

        data = extract_json(raw)
        items = data.get("标签体系", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            print("      [WARN] 本批输出结构不对，跳过")
            continue

        added = 0
        for it in items:
            tag_id = it.get("标签ID") if isinstance(it, dict) else None
            if not tag_id:
                continue
            if tag_id in seen_ids:
                continue
            seen_ids.add(tag_id)
            all_items.append(it)
            added += 1

        print(f"      本批新增 {added}，累计 {len(seen_ids)}")

    # 允许超出（模型多吐），但最终截断到 target_total
    all_items = all_items[:target_total]

    out_path = OUT / "02_meta_candidates.json"
    out_path.write_text(json.dumps({"标签体系": all_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[3/4] saved raw candidates:", out_path, "count=", len(all_items))

    # 生成后自动规范化（输出 normalized 文件）
    print("[4/4] normalize taxonomy ...")
    from scripts.normalize_taxonomy import main as norm_main
    norm_main()


if __name__ == "__main__":
    main()