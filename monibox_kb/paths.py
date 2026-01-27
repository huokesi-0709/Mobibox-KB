"""
paths.py
统一管理项目路径，彻底避免“相对路径依赖工作目录”的问题。

无论你在 PyCharm 点运行，还是在命令行运行，
都能正确找到 knowledge_src / sql / build 等目录。
"""
from pathlib import Path

# 当前文件：MoniBox-KB/monibox_kb/paths.py
# parents[1] => MoniBox-KB 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_SRC = PROJECT_ROOT / "knowledge_src"
GENERATED_DIR = KNOWLEDGE_SRC / "generated"

BUILD_DIR = PROJECT_ROOT / "build"
SQL_DIR = PROJECT_ROOT / "sql"