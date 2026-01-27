"""
config.py
统一读取 .env 配置。

关键点：
- 不再依赖“当前工作目录”来寻找 .env
- 显式从项目根目录加载 .env（paths.PROJECT_ROOT）
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from monibox_kb.paths import PROJECT_ROOT

# 显式从项目根目录加载 .env
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Settings:
    # DeepSeek (OpenAI兼容)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model_chat: str = os.getenv("DEEPSEEK_MODEL_CHAT", "deepseek-chat")
    deepseek_model_reasoner: str = os.getenv("DEEPSEEK_MODEL_REASONER", "deepseek-reasoner")

    # Embedding
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # Chunking
    chunk_max_chars: int = int(os.getenv("CHUNK_MAX_CHARS", "60"))
    chunk_min_chars: int = int(os.getenv("CHUNK_MIN_CHARS", "15"))

    # Build output
    rag_db_path: str = os.getenv("RAG_DB_PATH", "build/rag.db")
    runtime_pack_path: str = os.getenv("RUNTIME_PACK_PATH", "build/runtime_pack.json")


settings = Settings()