import hashlib
from monibox_kb.text_clean import clean_text

def sha256_fp(text: str) -> str:
    """
    对清洗后的文本做 sha256 指纹，用于严格去重。
    """
    t = clean_text(text)
    return "sha256:" + hashlib.sha256(t.encode("utf-8")).hexdigest()