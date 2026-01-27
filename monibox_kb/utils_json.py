"""
utils_json.py

目标：把 LLM 输出尽可能稳定地解析成 JSON（Python对象）
因为 LLM 常见问题：
- 输出带 ```json 围栏
- JSON 前后夹杂文字
- 输出被截断（缺少结尾的 ] 或 }）
- JSON 不严格（单引号、尾逗号、未加引号的 key）

策略：
1) 去围栏
2) 从文本中截取“第一个 { 或 [ 到最后一个 } 或 ]”的片段
3) 若括号未闭合：自动补齐缺失的 ] / }
4) strict json 解析失败就用 json5 兜底
"""

import json
import re
from typing import Any, Optional
import json5


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    # 去掉 ```json
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    # 去掉 ```
    t = re.sub(r"\s*```\s*$", "", t, flags=re.IGNORECASE)
    return t.strip()


def first_start(s: str) -> Optional[int]:
    for i, ch in enumerate(s):
        if ch in "{[":
            return i
    return None


def last_end(s: str) -> Optional[int]:
    for i in range(len(s) - 1, -1, -1):
        if s[i] in "}]":
            return i
    return None


def repair_unclosed_brackets(s: str) -> str:
    """
    自动补齐未闭合的括号。忽略字符串内部括号。
    例如：{"a":[{"b":1}   -> 自动补成 {"a":[{"b":1}]}
    """
    stack = []
    in_str = False
    escape = False

    for ch in s:
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                # 尝试正常出栈
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                else:
                    # 异常情况：括号类型不匹配，忽略（让 json5 去兜底）
                    pass

    # 补齐
    closing = []
    while stack:
        top = stack.pop()
        closing.append("}" if top == "{" else "]")
    return s + "".join(closing)


def extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM 返回空字符串（可能网络错误/被限流/请求失败）")

    t = strip_fences(raw)

    # 1) 先 strict 直接解析
    try:
        return json.loads(t)
    except Exception:
        pass

    # 2) 截取 JSON 大块
    st = first_start(t)
    ed = last_end(t)
    if st is None or ed is None or ed <= st:
        raise ValueError(f"无法定位JSON块。输出前200字符：{t[:200]!r}")

    block = t[st:ed + 1].strip()

    # 3) 尝试自动补齐括号（解决“截断”）
    block2 = repair_unclosed_brackets(block)

    # 4) strict 再试
    try:
        return json.loads(block2)
    except Exception:
        pass

    # 5) json5 兜底（允许单引号/尾逗号/未加引号 key）
    try:
        return json5.loads(block2)
    except Exception as e:
        raise ValueError(
            "JSON解析失败（strict json 与 json5 都失败）。\n"
            f"错误：{e}\n"
            f"候选JSON块前200字符：{block2[:200]!r}\n"
            f"候选JSON块后200字符：{block2[-200:]!r}"
        )