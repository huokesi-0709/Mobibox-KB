"""
generate_taxonomy.py（一键版：分页生成 + 自动规范化）

做什么：
1) 分批调用 DeepSeek 生成候选标签（避免一次太长被截断）
2) 去重合并后保存：knowledge_src/generated/02_meta_candidates.json
3) 自动运行本地规范化脚本 normalize_taxonomy.py
   输出：knowledge_src/generated/02_meta_candidates.normalized.json

你在 PyCharm 点绿色按钮运行这个文件即可完成 1~3。
"""

import sys
from pathlib import Path
import json

from monibox_kb.deepseek_client import DeepSeekClient
from monibox_kb.utils_json import extract_json
from monibox_kb.paths import KNOWLEDGE_SRC as SRC, GENERATED_DIR as OUT

# ---- 保险：即使你用“脚本方式”运行，也保证项目根目录在 sys.path ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    meta_path = SRC / "00_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 {meta_path}")

    print("[1/6] 读取 meta:", meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dims = meta["枚举"]["五维度"]

    # === 你关心的参数：可以按需要调整 ===
    target_total = 120   # 想最终得到多少个去重后的标签
    batch_size = 15      # 每次请求生成多少个（越大越容易截断）
    max_batches = 20     # 最大批次数（留余量防重复/防无限循环）

    cli = DeepSeekClient()
    all_items: list[dict] = []
    seen_ids: set[str] = set()

    print(f"[2/6] 开始分页生成：target_total={target_total}, batch_size={batch_size}, max_batches={max_batches}")

    batch_no = 0
    while len(seen_ids) < target_total and batch_no < max_batches:
        batch_no += 1
        need = min(batch_size, target_total - len(seen_ids))

        print(f"[3/6] Batch {batch_no}: 期望生成 {need} 个（已有 {len(seen_ids)}）...")
        raw = cli.chat(SYSTEM, build_prompt(dims, need, list(seen_ids)), temperature=0.6)

        raw_path = OUT / f"_debug_taxonomy_raw_batch{batch_no:02d}.txt"
        raw_path.write_text(raw or "", encoding="utf-8")
        print(f"      debug raw saved: {raw_path.name} (len={len(raw or '')})")

        data = extract_json(raw)
        if not isinstance(data, dict) or "标签体系" not in data:
            print("      [WARN] 输出不是预期结构，跳过本批次")
            continue

        items = data["标签体系"]
        if not isinstance(items, list):
            print("      [WARN] 标签体系不是数组，跳过本批次")
            continue

        added = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            tag_id = it.get("标签ID")
            if not tag_id or not isinstance(tag_id, str):
                continue
            if tag_id in seen_ids:
                continue
            seen_ids.add(tag_id)
            all_items.append(it)
            added += 1

        print(f"      本批新增：{added}，累计：{len(seen_ids)}")

    if len(seen_ids) == 0:
        raise RuntimeError("生成失败：没有得到任何可用标签。请查看 generated/_debug_taxonomy_raw_batch*.txt")

    # 截断到 target_total（防止模型多吐）
    all_items = all_items[:target_total]

    out_path = OUT / "02_meta_candidates.json"
    out_path.write_text(json.dumps({"标签体系": all_items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[4/6] saved raw candidates:", out_path, "count=", len(all_items))

    # 自动规范化（本地，不花钱）
    print("[5/6] normalize taxonomy (local) ...")
    from scripts.normalize_taxonomy import main as norm_main
    norm_main()

    print("[6/6] DONE. (raw + normalized are ready)")


if __name__ == "__main__":
    main()