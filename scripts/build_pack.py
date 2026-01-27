"""
build_pack.py
用途：
- 读取 12_chunks_synth.json
- 生成 embedding（bge-small-zh）
- 写入 SQLite + sqlite-vec（build/rag.db）
- 同时输出 runtime_pack.json（端侧运行时加载用）

注意：
- 这里只编译“合成 chunks”
- 你以后加入权威 chunks，只需要把它们合并成一个 chunks 列表再编译即可
"""
import json
from pathlib import Path

from monibox_kb.db_sqlitevec import RagDB
from monibox_kb.embedding import embed_texts
from monibox_kb.config import settings
from monibox_kb.paths import KNOWLEDGE_SRC as SRC, GENERATED_DIR as GEN, BUILD_DIR


def main():
    chunks_path = GEN / "12_chunks_synth.json"
    if not chunks_path.exists():
        raise FileNotFoundError("缺少 12_chunks_synth.json，请先运行 scripts/qa_to_chunks.py")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    # 生成向量（构建期在PC上做，端侧只负责检索）
    texts = [c["文本"] for c in chunks]
    vectors = embed_texts(texts)

    # 确保 build 目录存在
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 建库入库
    db = RagDB(settings.rag_db_path)
    db.create_tables()
    db.insert_chunks(chunks, vectors)

    # 生成 runtime_pack.json（先包含 meta + sources，后续你会加协议/意图/传感器事件）
    meta = json.loads((SRC / "00_meta.json").read_text(encoding="utf-8"))
    sources = json.loads((SRC / "01_sources.json").read_text(encoding="utf-8"))

    runtime_pack = {
        "格式版本": meta.get("格式版本", "1.0"),
        "嵌入模型": meta.get("嵌入模型"),
        "枚举": meta.get("枚举"),
        "标签体系": meta.get("标签体系"),
        "来源注册表": sources
    }

    Path(settings.runtime_pack_path).write_text(
        json.dumps(runtime_pack, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("built db:", settings.rag_db_path)
    print("built runtime_pack:", settings.runtime_pack_path)


if __name__ == "__main__":
    main()