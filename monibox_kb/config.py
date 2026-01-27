"""
config.py
统一读取 .env 配置，并把关键路径（rag.db / runtime_pack）解析成“项目根目录下的绝对路径”。

这样就不会受到 PyCharm 当前工作目录（cwd）影响。
"""
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

from monibox_kb.paths import PROJECT_ROOT

# 显式从项目根目录加载 .env
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(p: str) -> str:
    """
    把 .env 里的路径解析为绝对路径：
    - 若是相对路径：相对 PROJECT_ROOT
    - 若是绝对路径：原样返回
    """
    s = (p or "").strip()
    if not s:
        return s
    path = Path(s)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


@dataclass
class Settings:
    # DeepSeek
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model_chat: str = os.getenv("DEEPSEEK_MODEL_CHAT", "deepseek-chat")
    deepseek_model_reasoner: str = os.getenv("DEEPSEEK_MODEL_REASONER", "deepseek-reasoner")

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # Chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "60"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "15"))

    # Build output（关键：解析为绝对路径）
    rag_db_path: str = resolve_project_path(os.getenv("RAG_DB_PATH", "build/rag.db"))
    runtime_pack_path: str = resolve_project_path(os.getenv("RUNTIME_PACK_PATH", "build/runtime_pack.json"))


settings = Settings()