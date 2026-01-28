"""
generate_qa_fill_missing_tags.py (v2)
用途：自动补齐 “taxonomy 有但 db 没有(coverage=0)” 的标签样本，并支持：
- only_prefix：只补指定前缀（如 med_,env_,psy_）
- exclude_prefix：排除指定前缀（如 scn_,dev_,lang_）
- priority_prefix：优先补哪些前缀（如 med_,env_,psy_ 排在最前）
- dry_run：只打印将要补哪些标签，不调用 DeepSeek（省钱预览）

推荐运行：
  python -m scripts.generate_qa_fill_missing_tags --max_tags 12 --per_tag 5 --only_prefix "med_,env_,psy_"
  python -m scripts.generate_qa_fill_missing_tags --max_tags 12 --per_tag 5 --exclude_prefix "scn_,dev_,lang_"
  python -m scripts.generate_qa_fill_missing_tags --max_tags 12 --per_tag 5 --priority_prefix "med_,env_,psy_,act_"

补完后还需要：
  python -m scripts.qa_to_chunks
  python -m scripts.build_pack
  python -m scripts.tag_coverage_report
"""

import argparse
import json
import sqlite3
import uuid
from collections import Counter
from typing import Dict, List, Set, Any, Tuple

from monibox_kb.config import settings
from monibox_kb.paths import GENERATED_DIR as GEN, KNOWLEDGE_SRC as SRC
from monibox_kb.deepseek_client import DeepSeekClient
from monibox_kb.utils_json import extract_json
from monibox_kb.tags.registry import TagRegistry
import re

def _fix_common_json_issues(s: str) -> str:
    """
    针对 LLM 常见的 JSON 破坏做轻度修复：
    1) 把换行断开的 key 和 ':' 合并（"适用人群"\n: -> "适用人群":）
    2) 把多余的尾部括号裁剪掉（很多 ]} 重复）
    注意：这是“轻度修复”，不保证修复所有情况，但能覆盖你这次的错误。
    """
    if not s:
        return s

    # 去掉围栏
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s, flags=re.IGNORECASE)

    # 修复： "key"\n:  => "key":
    s = re.sub(r'"\s*\n\s*:\s*', '": ', s)

    # 修复：某些情况下 key 被拆成两行： "适用人群"\n : "xxx"
    s = re.sub(r'"\s*\n\s*"', '" "', s)  # 极少见，保险

    return s


def safe_extract_json(raw: str):
    """
    安全解析：先轻度修复，再 extract_json。
    若仍失败，返回 None（由调用方决定跳过/重试）。
    """
    from monibox_kb.utils_json import extract_json

    try:
        return extract_json(raw)
    except Exception:
        # 尝试一次轻度修复
        fixed = _fix_common_json_issues(raw)
        try:
            return extract_json(fixed)
        except Exception:
            return None


REQUIRED_KEYS = ["维度", "风险等级", "适用人群", "标签", "召回词", "问题", "回答", "安全提示"]


def parse_csv(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_tags_flat(tags_flat: str) -> List[str]:
    if not tags_flat:
        return []
    s = tags_flat.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [p.strip() for p in s.split("|") if p.strip()]


def load_db_tags(db_path: str) -> Counter:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT tags_flat FROM chunks").fetchall()
    conn.close()

    cnt = Counter()
    for r in rows:
        for t in parse_tags_flat(r["tags_flat"]):
            cnt[t] += 1
    return cnt


def load_normalized_taxonomy() -> List[Dict[str, Any]]:
    p = GEN / "02_meta_candidates.normalized.json"
    if not p.exists():
        raise FileNotFoundError(f"缺少 {p}，请先运行 taxonomy + normalize")
    obj = json.loads(p.read_text(encoding="utf-8"))
    items = obj.get("标签体系", [])
    if not isinstance(items, list):
        raise ValueError("normalized taxonomy 结构不对：标签体系不是数组")
    return items


def build_prompt(tag_id: str, tag_name: str, dim: str, risk_hint: str, per_tag: int) -> str:
    return f"""
你是 MoniBox（地震受困陪伴设备）的问答数据生成器。
请生成 {per_tag} 条“问答数据”（JSON数组），全部围绕标签：{tag_id}（{tag_name}）。

硬性要求：
1) 只能输出严格 JSON 数组，不要解释，不要 Markdown，不要代码块。
2) 每条必须包含字段：
   qa_id, 维度, 子主题, 风险等级, 适用人群, 标签, 召回词, 问题, 回答, 安全提示
3) “维度”必须设为：{dim}
4) “标签”必须是数组，并且必须包含：{tag_id}
5) “风险等级”建议为：{risk_hint}
6) 回答必须：短句、可TTS播报、非诊断性、不提供剂量、不承诺获救时间。
7) 医学相关只能给自我保护/节省体力/合理求救/等待救援建议，禁止高风险医疗操作指导。
8) 召回词尽量覆盖口语表达、常见说法、错别字（最多10个）。

请直接输出 JSON 数组：
""".strip()


def starts_with_any(s: str, prefixes: List[str]) -> bool:
    return any(s.startswith(p) for p in prefixes)


def prefix_rank(tag_id: str, priority_prefixes: List[str]) -> int:
    """
    priority_prefixes 越靠前优先级越高（rank越小越优先）
    未命中任何优先前缀则排到后面。
    """
    for i, p in enumerate(priority_prefixes):
        if tag_id.startswith(p):
            return i
    return 999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tags", type=int, default=10, help="本次最多补多少个缺失标签（默认10）")
    ap.add_argument("--per_tag", type=int, default=4, help="每个标签生成多少条 QA（默认4）")

    # 旧开关：是否补 dev_/lang_
    ap.add_argument("--include_dev", action="store_true", help="是否补 dev_ 设备交互类标签（默认不补）")
    ap.add_argument("--include_lang", action="store_true", help="是否补 lang_ 方言类标签（默认不补）")

    # 新增：前缀过滤/优先级
    ap.add_argument("--only_prefix", default=None,
                    help="只补这些前缀（逗号分隔），例如 'med_,env_,psy_'")
    ap.add_argument("--exclude_prefix", default=None,
                    help="排除这些前缀（逗号分隔），例如 'scn_,dev_,lang_'")
    ap.add_argument("--priority_prefix", default=None,
                    help="优先补这些前缀（逗号分隔），例如 'med_,env_,psy_,act_'。不填则按 taxonomy 原顺序")

    ap.add_argument("--dry_run", action="store_true",
                    help="只打印将要补哪些标签，不调用 DeepSeek（省钱预览）")

    args = ap.parse_args()

    only_prefixes = parse_csv(args.only_prefix)
    exclude_prefixes = parse_csv(args.exclude_prefix)
    priority_prefixes = parse_csv(args.priority_prefix)

    # 1) taxonomy
    items = load_normalized_taxonomy()
    tax = []
    for idx, it in enumerate(items):
        tid = str(it.get("标签ID", "")).strip()
        if not tid:
            continue
        tax.append({
            "idx": idx,  # 保留原顺序
            "tag_id": tid,
            "name": str(it.get("名称", "")).strip(),
            "category": str(it.get("类别", "")).strip(),
            "dimension": str(it.get("所属维度", "")).strip(),
            "risk_hint": str(it.get("风险等级建议", "中")).strip() or "中"
        })

    # 2) db coverage
    db_cnt = load_db_tags(settings.rag_db_path)

    # 3) missing = coverage 0
    missing = [t for t in tax if db_cnt.get(t["tag_id"], 0) == 0]

    # 4) basic filtering: dev_/lang_ 默认不补
    filtered = []
    for t in missing:
        tid = t["tag_id"]

        if tid.startswith("dev_") and not args.include_dev:
            continue
        if tid.startswith("lang_") and not args.include_lang:
            continue

        if only_prefixes and not starts_with_any(tid, only_prefixes):
            continue

        if exclude_prefixes and starts_with_any(tid, exclude_prefixes):
            continue

        filtered.append(t)

    # 5) selection ordering
    if priority_prefixes:
        # 按优先前缀排序，再按原 taxonomy 顺序
        filtered.sort(key=lambda x: (prefix_rank(x["tag_id"], priority_prefixes), x["idx"]))
    else:
        # 默认按原 taxonomy 顺序
        filtered.sort(key=lambda x: x["idx"])

    selected = filtered[:max(0, int(args.max_tags))]

    print("==== generate_qa_fill_missing_tags (v2) ====")
    print("[info] db:", settings.rag_db_path)
    print("[info] taxonomy tags:", len(tax))
    print("[info] db tags:", len(db_cnt))
    print("[info] missing tags:", len(missing))
    print("[info] after filter:", len(filtered))
    print("[info] selected to fill:", len(selected))
    print("[info] only_prefix:", only_prefixes)
    print("[info] exclude_prefix:", exclude_prefixes)
    print("[info] priority_prefix:", priority_prefixes)

    for i, t in enumerate(selected[:50], start=1):
        print(f"  {i}. {t['tag_id']}  dim={t['dimension']} risk_hint={t['risk_hint']} name={t['name']}")

    if not selected:
        print("Nothing to fill. Done.")
        return

    if args.dry_run:
        print("\n[dry_run] stop here (no API call).")
        return

    # 6) load existing QA
    qa_path = GEN / "11_qa_synth.json"
    qa_all: List[Dict[str, Any]] = []
    seen = set()

    if qa_path.exists():
        old = json.loads(qa_path.read_text(encoding="utf-8"))
        if isinstance(old, list):
            qa_all = old
            for it in qa_all:
                q = str(it.get("问题", "")).strip()
                a = str(it.get("回答", "")).strip()
                if q and a:
                    seen.add(q + "\n" + a)

    # 7) TagRegistry (alias + canonicalize)
    reg = TagRegistry.load()

    cli = DeepSeekClient()

    for idx, t in enumerate(selected, start=1):
        tag_id = t["tag_id"]
        prompt = build_prompt(tag_id, t["name"], t["dimension"], t["risk_hint"], int(args.per_tag))

        print(f"\n[gen] ({idx}/{len(selected)}) tag={tag_id} per_tag={args.per_tag}")
        raw = cli.chat("只输出严格JSON数组，不要任何解释。", prompt, temperature=0.7)

        raw_path = GEN / f"_debug_qa_fill_{idx:02d}_{tag_id}.txt"
        raw_path.write_text(raw or "", encoding="utf-8")

        data = safe_extract_json(raw)
        if data is None:
            print("  [WARN] JSON 解析失败，已跳过该 tag（raw 已保存到 debug 文件）")
            continue


        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            print("  [WARN] 输出不是数组，跳过该 tag")
            continue

        added = 0
        for it in data:
            if not isinstance(it, dict):
                continue
            if any(k not in it for k in REQUIRED_KEYS):
                continue

            # 标签强制包含当前 tag + 做收敛
            tags = it.get("标签", [])
            if not isinstance(tags, list):
                tags = [tags]
            tags = [str(x) for x in tags if str(x).strip()]
            if tag_id not in tags:
                tags.append(tag_id)

            canon_tags = reg.canonicalize_list(tags)
            if not canon_tags:
                continue
            it["标签"] = canon_tags

            q = str(it.get("问题", "")).strip()
            a = str(it.get("回答", "")).strip()
            if not q or not a:
                continue
            fp = q + "\n" + a
            if fp in seen:
                continue
            seen.add(fp)

            it["qa_id"] = "qa_" + uuid.uuid4().hex[:12]
            it["来源ID"] = "src_synth_deepseek_v1"
            it["状态"] = "候选"
            it["人工评分"] = 0

            qa_all.append(it)
            added += 1

        print(f"  added={added}")

    qa_path.write_text(json.dumps(qa_all, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDONE. updated:", qa_path, "total QA:", len(qa_all))
    print("Next:")
    print("  python -m scripts.qa_to_chunks")
    print("  python -m scripts.build_pack")
    print("  python -m scripts.tag_coverage_report")


if __name__ == "__main__":
    main()