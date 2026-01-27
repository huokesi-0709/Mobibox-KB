"""
cleanup_generated.py
用途：清理 generated 目录中历史调试文件（attempt/batch的 raw 输出）
这些文件只用于排查模型输出，正常跑通后可以删掉，目录会更干净。
"""

from monibox_kb.paths import GENERATED_DIR as GEN

PATTERNS = [
    "_debug_taxonomy_raw_attempt",
    "_debug_taxonomy_raw.txt",
    "_debug_qa_raw_attempt",
]

def main():
    removed = 0
    for p in GEN.iterdir():
        if not p.is_file():
            continue
        name = p.name
        # 清理 attempt 版（你现在已经不用了）
        if any(name.startswith(prefix) for prefix in PATTERNS):
            p.unlink(missing_ok=True)
            removed += 1

    print("DONE. removed files:", removed)

if __name__ == "__main__":
    main()