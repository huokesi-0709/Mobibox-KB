"""
monibox_kb/runtime/session.py

用途
-----
MoniSession：Windows/Radxa 通用会话编排层（端到端链路核心）
实现：用户输入（文本/ASR） -> 路由 -> 协议优先 -> RAG 检索 -> LLM(基于RAG改写/整合) -> 安全护栏 -> TTS

关键设计点
---------
1) 协议优先：高风险/事件触发时，不依赖 LLM 自由生成
2) RAG + LLM：LLM 只做“基于 chunks 的改写/整合”，不得编造
3) LLM 输出强制 JSON：
   {"text": "...", "used_ids": ["..."], "ask": "..."}
   - text：给 TTS 播放（强制 <= 60字）
   - used_ids：调试/评分闭环（不播报）
   - ask：可选澄清问题（也会被纳入 <= 60字 控制）

工程稳态措施
------------
- 对 LLM stream 增加 stop：阻止输出第二个 JSON（常见模式是 \n{ 开始第二个对象）
- 使用 extract_first_json：即使模型输出多个 JSON，也能解析第一个完整对象
- 60字硬限制：不要指望模型自觉，必须后处理强制截断
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Any, Dict

from monibox_kb.runtime.rag_engine import RagEngine
from monibox_kb.runtime.protocol_engine import ProtocolEngine
from monibox_kb.runtime.safety_guard import SafetyGuard
from monibox_kb.llm.llama_cpp_chat import LLMConfig, LlamaCppChat
from monibox_kb.tts.pyttsx3_tts import Pyttsx3TTS
from monibox_kb.utils_json import extract_first_json


# -----------------------------
# Prompt 模板（Qwen 灾害场景）
# -----------------------------
def build_system_prompt() -> str:
    """
    system prompt：固定角色、安全边界、输出格式（强制 JSON）
    建议尽量少改这个，主要靠 RAG/评分闭环提升内容质量。
    """
    return (
        "你是灾害受困陪伴设备 MoniBox，目标：稳定情绪、节省体力、保护自身、等待救援。\n"
        "安全要求：不做诊断；不提供药物剂量；不指导侵入性医疗操作；不建议危险自救。\n"
        "回答风格：中文、镇定、短句、先给最关键下一步行动。\n"
        "\n"
        "你会得到：用户话语 + 已检索到的知识要点（每条含 id 与 text）。\n"
        "规则：\n"
        "1) 只能依据“已检索要点”作答，不得编造新事实。\n"
        "2) 如果要点不足以支撑回答：先给一个安全的通用动作，再 ask 一个澄清问题。\n"
        "3) 你必须只输出【一个】JSON对象，输出后立刻停止，不要再输出第二个JSON，不要解释。\n"
        "\n"
        "JSON 格式（必须包含这3个 key）：\n"
        "{"
        '"text":"(给用户的短回复，<=60字，最多2句)",'
        '"used_ids":["(使用到的要点id)"],'
        '"ask":"(可选澄清问题，<=60字；不需要则为空字符串)"'
        "}\n"
    )


def build_user_prompt(user_text: str, retrieved_items: List[Dict[str, str]]) -> str:
    """
    user prompt：把 RAG 命中的片段（含 id + text）传给 LLM
    retrieved_items: [{"id": "...", "text": "..."}, ...]
    """
    lines = []
    for it in retrieved_items[:6]:
        cid = (it.get("id") or "").strip()
        txt = (it.get("text") or "").strip()
        if txt:
            lines.append(f'- id="{cid}" text="{txt}"')

    ctx = "\n".join(lines) if lines else "(无)"

    return (
        f"已检索要点：\n{ctx}\n\n"
        f"用户：{user_text}\n"
    )


# -----------------------------
# 文本后处理（TTS友好）
# -----------------------------
def normalize_for_tts(text: str) -> str:
    """清理换行/多空格，让 TTS 更稳定。"""
    t = (text or "").strip()
    t = t.replace("\r", " ").replace("\n", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t.strip()


def limit_chars(text: str, max_chars: int = 60) -> str:
    """
    限制最大字符数（中文按 len 计即可）。
    这里硬截断，不加省略号，避免 TTS 读“省略号”影响节奏。
    """
    t = normalize_for_tts(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars].strip()


def parse_llm_payload(raw: str) -> Dict[str, Any]:
    """
    将 LLM 输出解析为 payload dict：
    - 优先用 extract_first_json（支持多 JSON 连续输出）
    - 失败则降级为 {"text": raw, "used_ids": [], "ask": ""}

    返回字段保证存在：text(str), used_ids(list[str]), ask(str)
    """
    raw = (raw or "").strip()
    if not raw:
        return {"text": "", "used_ids": [], "ask": ""}

    try:
        obj = extract_first_json(raw)
        if not isinstance(obj, dict):
            return {"text": raw, "used_ids": [], "ask": ""}

        text = obj.get("text", "") or ""
        used_ids = obj.get("used_ids", []) or []
        ask = obj.get("ask", "") or ""

        if isinstance(used_ids, str):
            used_ids = [used_ids]
        if not isinstance(used_ids, list):
            used_ids = []

        return {"text": str(text), "used_ids": used_ids, "ask": str(ask)}
    except Exception:
        return {"text": raw, "used_ids": [], "ask": ""}


@dataclass
class SessionConfig:
    llm_path: str
    llm_ctx: int = 2048
    llm_threads: int = 6
    llm_gpu_layers: int = 0
    tts_enabled: bool = True


class MoniSession:
    """
    Windows/Radxa 通用会话：
    - 协议优先（余震/粉尘/恐慌）
    - 未命中协议：RAG + LLM（流式）生成最终回复
    - 安全护栏：协议与LLM输出都过滤
    - TTS：Windows 用 pyttsx3（Radxa 后续可替换）
    """

    def __init__(self, rag_db_path: str, cfg: SessionConfig):
        self.rag = RagEngine(rag_db_path)
        self.prot = ProtocolEngine()
        self.guard = SafetyGuard()

        self.tts_enabled = cfg.tts_enabled
        self.tts = Pyttsx3TTS(
            rate=int(os.getenv("TTS_RATE", "180")),
            volume=float(os.getenv("TTS_VOLUME", "1.0")),
        )

        llm_cfg = LLMConfig(
            gguf_path=cfg.llm_path,
            n_ctx=cfg.llm_ctx,
            n_threads=cfg.llm_threads,
            n_gpu_layers=cfg.llm_gpu_layers,
        )
        self.llm = LlamaCppChat(llm_cfg)

        # stop：防止模型输出第二个 JSON（经验上最常见的第二个对象起始是换行 + '{'）
        self.llm_stop = ["\n{", "\r\n{", "</s>", "<|endoftext|>"]

    def _speak(self, text: str):
        if self.tts_enabled and text:
            self.tts.speak(text)

    def handle(self, user_text: str, events: Optional[List[str]] = None, auto_top_tags: int = 2) -> str:
        events = events or []
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        # 1) 路由标签（用于协议触发 + RAG过滤）
        rr = self.rag.router.route(user_text, top_tags=auto_top_tags)

        # 2) 协议优先
        hit = self.prot.match(user_text, rr.tags, events)
        if hit:
            out_lines = []
            for a in hit.get("actions", []):
                if a.get("type") == "tts":
                    gr = self.guard.check(a.get("text", ""))
                    out_lines.append(gr.safe_text)

            final = normalize_for_tts("\n".join([x for x in out_lines if x]).strip())
            final = limit_chars(final, 60)

            print("\n[PROTOCOL HIT]", hit.get("protocol_id"), hit.get("name"))
            print(final)
            self._speak(final)
            return final

        # 3) RAG 检索
        dim = None if rr.cross_dimension else rr.dimension
        results = self.rag.search(
            user_text,
            topk=6,
            pool_mult=8,
            dimension=dim,
            tags=rr.tags,
            max_per_group=1,
        )

        # 给 LLM 的上下文：带 id + text（用于“引用不编造”与评分闭环）
        retrieved_items: List[Dict[str, str]] = []
        for r in results:
            cid = getattr(r, "display_id", None) or getattr(r, "chunk_id", None) or ""
            retrieved_items.append({"id": str(cid), "text": r.text})

        # 如果 RAG 完全没命中，也允许 LLM 做“通用安全动作 + 澄清问题”
        system = build_system_prompt()
        user = build_user_prompt(user_text, retrieved_items)

        # 4) LLM 流式生成（要求输出 JSON）
        print("\n[NO PROTOCOL] RAG+LLM streaming(JSON)...")
        buf = ""
        for tok in self.llm.stream_chat(
            system,
            user,
            max_tokens=220,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            top_p=float(os.getenv("LLM_TOP_P", "0.9")),
            stop=self.llm_stop,
        ):
            buf += tok
            print(tok, end="", flush=True)
        print("\n")

        # 5) 解析 JSON（失败则降级）
        payload = parse_llm_payload(buf)

        text = (payload.get("text") or "").strip()
        ask = (payload.get("ask") or "").strip()
        used_ids = payload.get("used_ids") or []

        # 6) 合成最终给 TTS 的文本（<=60字）
        merged = text
        if ask:
            merged = (merged + " " + ask).strip()

        merged = limit_chars(merged, 60)

        # 7) 安全护栏（最终回复）
        gr = self.guard.check(merged)
        final = normalize_for_tts(gr.safe_text.strip())
        final = limit_chars(final, 60)

        # 8) 打印调试信息（不播报）：used_ids 便于评分闭环/命中追溯
        if used_ids:
            print("[LLM USED_IDS]", used_ids)

        self._speak(final)
        return final