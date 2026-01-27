import re

def clean_text(s: str) -> str:
    """
    基础清洗：空白规范化。
    你后续可以加入：全角半角统一、敏感信息剔除等。
    """
    s = re.sub(r"\s+", " ", s).strip()
    return s