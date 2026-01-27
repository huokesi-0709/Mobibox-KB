"""
build_pack.py（可视化增强版 v2）

新增能力：
1) 每次构建前自动删除旧 rag.db（避免上次失败留下半成品导致 UNIQUE 冲突）
2) 构建前检查 chunks 中片段ID是否重复（若重复会输出报告并直接停止）
3) 日志更清晰：你知道它做到了哪一步

说明：
- 你现在处于调试阶段，rag.db 本来就是可删可重建的构建产物
- 所以这里采用“强制重建策略”：每次都删掉旧库重新建
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

from monibox_kb.config import settings
from monibox_kb.paths import KNOWLEDGE_SRC as SRC, GENERATED_DIR as GEN, BUILD_DIR, PROJECT_ROOT
from monibox_kb.db_sqlitevec import RagDB
from monibox_kb.embedding import embed_texts, get_model


REQUIRED_FIELDS = ["片段ID", "文本", "维度", "风险等级", "来源ID", "状态", "内容指纹"]


def human_bytes(n: int) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < step:
            return f"{x:.2f}{u}"
        x /= step
    return f"{x:.2f}PB"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_chunks(chunks: List[Dict[str, Any]], dup_report_path: Path) -> None:
    """
    校验：
    1) 必填字段存在
    2) 文本非空
    3) 片段ID唯一（否则 sqlite UNIQUE 必炸）
    """
    if not isinstance(chunks, list):
        raise ValueError("chunks 文件结构不对：根必须是数组")

    # 先查重复ID
    ids = [c.get("片段ID") for c in chunks if isinstance(c, dict)]
    cnt = Counter(ids)
    dups = [k for k, v in cnt.items() if k and v > 1]
    if dups:
        # 输出重复报告，方便你定位是哪几条
        report = {
            "duplicate_count": len(dups),
            "duplicates": dups[:50]  # 前50个
        }
        dup_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(
            f"发现重复片段ID（数量={len(dups)}），已写入报告：{dup_report_path}\n"
            f"示例：{dups[:5]}"
        )

    # 再做字段校验
    for i, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"chunks[{i}] 不是对象 dict")

        miss = [k for k in REQUIRED_FIELDS if k not in c]
        if miss:
            raise ValueError(f"chunks[{i}] 缺字段：{miss}。片段ID={c.get('片段ID')}")

        if not isinstance(c.get("文本"), str) or not c["文本"].strip():
            raise ValueError(f"chunks[{i}] 文本为空。片段ID={c.get('片段ID')}")

        if "标签" in c and c["标签"] is not None and not isinstance(c["标签"], list):
            raise ValueError(f"chunks[{i}] 标签必须是数组 list。片段ID={c.get('片段ID')}")
        if "适用人群" in c and c["适用人群"] is not None and not isinstance(c["适用人群"], list):
            raise ValueError(f"chunks[{i}] 适用人群必须是数组 list。片段ID={c.get('片段ID')}")


def main():
    print("==== MoniBox-KB build_pack.py ====")
    print("[info] project root:", PROJECT_ROOT)
    print("[info] python cwd:", os.getcwd())
    print("[info] rag db path:", settings.rag_db_path)
    print("[info] runtime pack path:", settings.runtime_pack_path)
    print()

    chunks_path = GEN / "12_chunks_synth.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"缺少 {chunks_path}\n请先运行：python scripts/qa_to_chunks.py"
        )

    print("[1/8] 读取 chunks 文件:", chunks_path)
    chunks = load_json(chunks_path)
    print(f"      chunks loaded: {len(chunks)} 条")

    print("[2/8] 校验 chunks（含重复片段ID检查）...")
    dup_report_path = GEN / "12_chunks_duplicate_report.json"
    validate_chunks(chunks, dup_report_path)
    print("      ok")

    # 确保 build 目录存在
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 关键：调试阶段强制删旧库，避免半成品导致 UNIQUE 冲突
    db_path = Path(settings.rag_db_path)
    if db_path.exists():
        print("[3/8] 检测到旧 rag.db，删除以重建：", db_path)
        db_path.unlink()

    print("[4/8] 加载 embedding 模型（本地）...")
    model = get_model()  # embedding.py 会打印本地路径
    print("      embedding model loaded:", type(model))

    texts = [c["文本"] for c in chunks]
    print("[5/8] 生成向量 embedding ...")
    vectors = embed_texts(texts)
    vec_dim = len(vectors[0]) if vectors else 0
    print(f"      embedding done. vectors={len(vectors)} dim={vec_dim}")

    print("[6/8] 创建/初始化数据库并写入 ...")
    db = RagDB(settings.rag_db_path)
    db.create_tables()
    db.insert_chunks(chunks, vectors)
    print("      db insert done.")

    print("[7/8] 生成 runtime_pack.json ...")
    meta = load_json(SRC / "00_meta.json")
    sources = load_json(SRC / "01_sources.json")
    runtime_pack = {
        "格式版本": meta.get("格式版本", "1.0"),
        "嵌入模型": meta.get("嵌入模型"),
        "枚举": meta.get("枚举"),
        "标签体系": meta.get("标签体系"),
        "来源注册表": sources
    }
    out_pack = Path(settings.runtime_pack_path)
    out_pack.parent.mkdir(parents=True, exist_ok=True)
    out_pack.write_text(json.dumps(runtime_pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print("      runtime_pack saved:", out_pack)

    print("[8/8] 数据库统计信息 ...")
    size = db_path.stat().st_size if db_path.exists() else 0
    print("      rag.db:", db_path)
    print("      size:", human_bytes(size))
    print("==== DONE ====")


if __name__ == "__main__":
    main()