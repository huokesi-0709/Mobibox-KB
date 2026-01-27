"""
monibox_kb/embedding.py

作用：
- 在构建期（PC）为文本生成 embedding 向量，用于写入 sqlite-vec（rag.db）
- 支持使用“本地模型目录”，避免重复下载

你现在已把模型放在：
D:\\代码项目\\MoniBox-KB\\models\\embedding\\bge-small-zh-v1.5

推荐做法：
- 在 .env 里配置 EMBEDDING_MODEL=models/embedding/bge-small-zh-v1.5
- 本文件会自动把相对路径拼成项目根目录下的绝对路径
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from monibox_kb.config import settings
from monibox_kb.paths import PROJECT_ROOT


_model: SentenceTransformer | None = None


def _resolve_model_ref(model_id_or_path: str) -> str:
    """
    把 .env 中的 EMBEDDING_MODEL 解析为真正描述符：
    - 如果是相对路径：按 PROJECT_ROOT 拼成绝对路径
    - 如果是绝对路径：直接使用
    - 如果是 HuggingFace 模型名：原样返回（但你现在不需要）
    """
    s = (model_id_or_path or "").strip()
    if not s:
        raise ValueError("EMBEDDING_MODEL 为空，请在 .env 中设置")

    p = Path(s)

    # 相对路径：相对于项目根目录
    if not p.is_absolute() and ("/" in s or "\\" in s):
        p = (PROJECT_ROOT / p).resolve()
        return str(p)

    # 绝对路径
    if p.is_absolute():
        return str(p)

    # 不是路径：当作模型名（不建议你现在用）
    return s


def get_model() -> SentenceTransformer:
    """
    单例加载 embedding 模型，避免重复加载占内存。
    """
    global _model
    if _model is None:
        model_ref = _resolve_model_ref(settings.embedding_model)

        # 强制离线（可选）：如果你担心它去联网，可以打开下面两行
        # 由于你提供的是“本地路径”，正常情况下不会触发下载；
        # 但开启离线更保险。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # 给出明确提示，方便你确认它在用本地路径
        print(f"[embedding] loading model from: {model_ref}")

        # 检查关键文件是否存在，避免 silent fail
        if Path(model_ref).exists():
            cfg = Path(model_ref) / "config_sentence_transformers.json"
            if not cfg.exists():
                print("[embedding][WARN] 未发现 config_sentence_transformers.json，"
                      "如果加载失败，请确认该目录是 sentence-transformers 格式模型。")

        _model = SentenceTransformer(model_ref)

    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    批量生成向量（normalize_embeddings=True 输出单位向量，适合余弦相似度）
    """
    model = get_model()
    emb = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb.tolist()