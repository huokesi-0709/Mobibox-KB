"""
qa_to_chunks.py
用途：
- 把 QA 的“回答”切成 60 字左右的原子片段（TTS友好）
- 自动添加：来源ID / 状态 / 人工评分 / 内容指纹（用于去重与可追溯）
- 输出：knowledge_src/generated/12_chunks_synth.json
"""
import json
from monibox_kb.config import settings
from monibox_kb.chunking import split_by_max_chars
from monibox_kb.text_clean import clean_text
from monibox_kb.dedup import sha256_fp
from monibox_kb.paths import GENERATED_DIR as GEN

OUT_PATH = GEN / "12_chunks_synth.json"


def main():
    qa_path = GEN / "11_qa_synth.json"
    if not qa_path.exists():
        raise FileNotFoundError("缺少 11_qa_synth.json，请先运行 scripts/generate_qa.py")

    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    chunks = []
    for it in qa:
        answer = clean_text(it["回答"])
        atoms = split_by_max_chars(answer, settings.chunk_max_chars, settings.chunk_min_chars)

        # 把“召回词 + 问题”也塞进召回词集合，提高检索命中率
        recall_terms = set(it.get("召回词", []))
        if it.get("问题"):
            recall_terms.add(it["问题"])

        for i, atom in enumerate(atoms):
            chunk = {
                "类型": "知识片段",
                "片段ID": f'k_{it["qa_id"]}_{i:02d}',
                "维度": it["维度"],
                "子主题": it.get("子主题"),
                "风险等级": it["风险等级"],
                "适用人群": it.get("适用人群", []),
                "标签": it.get("标签", []),
                "召回词": sorted([x for x in recall_terms if x])[:80],

                "可直接播报": True,
                "播报风格": "冷静清晰",
                "文本": atom,

                # 你后续引入专家/权威数据时，只要来源ID不同、权威更高即可替换/重排
                "来源ID": it.get("来源ID", "src_synth_deepseek_v1"),
                "状态": it.get("状态", "候选"),
                "人工评分": it.get("人工评分", 0),
                "内容指纹": sha256_fp(atom),

                "__安全提示": it.get("安全提示", "")
            }
            chunks.append(chunk)

    OUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUT_PATH, "count=", len(chunks))


if __name__ == "__main__":
    main()