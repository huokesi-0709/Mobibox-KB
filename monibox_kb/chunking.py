import re
from typing import List

PUNCT = "。！？；…\n"

def split_by_max_chars(text: str, max_chars: int, min_chars: int) -> List[str]:
    """
    按最大字数切分，并尽量优先在中文标点处分割。
    适合 TTS 秒级播报。
    """
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []

    out = []
    i, n = 0, len(t)

    while i < n:
        j = min(i + max_chars, n)
        cut = j

        if j < n:
            # 从 j 往回找标点，避免切断句子
            for k in range(j, i + min_chars - 1, -1):
                if t[k - 1] in PUNCT:
                    cut = k
                    break

        piece = t[i:cut].strip()
        if piece:
            out.append(piece)
        i = cut

    return out