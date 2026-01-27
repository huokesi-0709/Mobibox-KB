"""
构建期 embedding（PC 上跑）。
Radxa 端也能跑，但速度慢；通常只在构建期生成向量并入库。
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from monibox_kb.config import settings

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    # normalize_embeddings=True 会输出单位向量，便于余弦相似度
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    return emb.tolist()