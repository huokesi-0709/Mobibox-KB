from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from monibox_kb.runtime.rag_engine import RagEngine
from monibox_kb.runtime.protocol_engine import ProtocolEngine
from monibox_kb.runtime.safety_guard import SafetyGuard
from monibox_kb.llm.llama_cpp_chat import LLMConfig, LlamaCppChat
from monibox_kb.tts.pyttsx3_tts import Pyttsx3TTS


def build_prompt(user_text: str, retrieved_chunks: List[str]) -> str:
    ctx = "\n".join([f"- {c}" for c in retrieved_chunks[:6]])
    return f"""你是灾害受困陪伴设备 MoniBox。目标：稳定情绪、节省体力、保护自身、等待救援。
要求：
- 输出中文，短句，先给最关键的下一步行动。
- 不做诊断，不给剂量，不指导侵入性医疗操作。
- 语气镇定、具体可执行。
- 如果信息不足，优先提出一个简单澄清问题。

已检索到的要点：
{ctx}

用户：{user_text}
MoniBox："""


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
        self.tts = Pyttsx3TTS(rate=int(os.getenv("TTS_RATE", "180")),
                              volume=float(os.getenv("TTS_VOLUME", "1.0")))

        llm_cfg = LLMConfig(
            gguf_path=cfg.llm_path,
            n_ctx=cfg.llm_ctx,
            n_threads=cfg.llm_threads,
            n_gpu_layers=cfg.llm_gpu_layers,
        )
        self.llm = LlamaCppChat(llm_cfg)

    def _speak(self, text: str):
        if self.tts_enabled:
            self.tts.speak(text)

    def handle(self, user_text: str, events: Optional[List[str]] = None, auto_top_tags: int = 2) -> str:
        events = events or []

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

            final = "\n".join([x for x in out_lines if x]).strip()
            print("\n[PROTOCOL HIT]", hit["protocol_id"], hit["name"])
            print(final)
            self._speak(final)
            return final

        # 3) RAG 检索（跨维度不锁 dimension）
        dim = None if rr.cross_dimension else rr.dimension
        results = self.rag.search(
            user_text,
            topk=6,
            pool_mult=8,
            dimension=dim,
            tags=rr.tags,
            max_per_group=1
        )
        retrieved = [r.text for r in results]

        # 4) LLM 流式生成（控制台流式显示）
        prompt = build_prompt(user_text, retrieved)
        print("\n[NO PROTOCOL] RAG+LLM streaming...")
        buf = ""
        for tok in self.llm.stream(prompt, max_tokens=220, temperature=0.6):
            buf += tok
            print(tok, end="", flush=True)
        print("\n")

        # 5) 安全护栏（最终回复）
        gr = self.guard.check(buf.strip())
        final = gr.safe_text.strip()

        self._speak(final)
        return final