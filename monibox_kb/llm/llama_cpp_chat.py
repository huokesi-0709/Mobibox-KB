"""
monibox_kb/llm/llama_cpp_chat.py

用途
-----
对 llama-cpp-python 的最小封装，让 MoniBox-KB 在 Windows/Radxa 上以统一接口调用本地 GGUF 量化模型：
- 支持 Chat Completions（system/user messages）
- 支持流式输出（stream），用于“边生成边在控制台显示”，最终再交给 TTS 播放
- 针对 Qwen1.5-Chat：默认使用 ChatML 模板（可通过环境变量覆盖）

如何使用
--------
1) .env 里配置：
   LLM_GGUF_PATH=models/llm/qwen1_5-0_5b-chat-q4_k_m.gguf
   LLM_CHAT_FORMAT=chatml   # 可选；Qwen1.5-Chat 通常用 chatml

2) 代码里：
   cfg = LLMConfig(gguf_path="models/llm/xxx.gguf", n_ctx=2048, n_threads=6, n_gpu_layers=0)
   llm = LlamaCppChat(cfg)

   system = "你是..."
   user = "用户：... 已检索要点：..."
   for tok in llm.stream_chat(system, user):
       print(tok, end="", flush=True)
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Optional

from llama_cpp import Llama

from monibox_kb.paths import PROJECT_ROOT


@dataclass
class LLMConfig:
    """LLM 配置（与 .env 对齐）"""
    gguf_path: str
    n_ctx: int = 2048
    n_threads: int = 6
    n_gpu_layers: int = 0
    chat_format: str = "chatml"  # Qwen1.5-Chat 通常用 chatml


class LlamaCppChat:
    """
    llama-cpp-python 封装：优先使用 Chat Completions（messages）接口。
    """

    def __init__(self, cfg: LLMConfig):
        p = Path(cfg.gguf_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到 GGUF 模型文件：{p}")

        # 允许环境变量覆盖（便于快速切换模板）
        chat_format = (os.getenv("LLM_CHAT_FORMAT", cfg.chat_format) or "").strip() or cfg.chat_format

        self.llm = Llama(
            model_path=str(p),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            chat_format=chat_format,
            verbose=False,
        )
        self.chat_format = chat_format

    @staticmethod
    def _messages(system: str, user: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if (system or "").strip():
            msgs.append({"role": "system", "content": system.strip()})
        msgs.append({"role": "user", "content": (user or "").strip()})
        return msgs

    def stream_chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.4,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """
        Chat 流式输出：逐段 yield token(string)
        """
        messages = self._messages(system, user)
        stop = stop or ["</s>", "<|endoftext|>"]

        sig = inspect.signature(self.llm.create_chat_completion)
        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        if "stop" in sig.parameters:
            kwargs["stop"] = stop

        for chunk in self.llm.create_chat_completion(**kwargs):
            # 更鲁棒的 token 提取：兼容 delta/content 或 text
            choice = (chunk.get("choices") or [{}])[0]
            tok = ""
            if isinstance(choice, dict):
                if "delta" in choice and isinstance(choice["delta"], dict):
                    tok = choice["delta"].get("content", "") or ""
                if not tok:
                    tok = choice.get("text", "") or ""
            if tok:
                yield tok

    def generate_chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.4,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Chat 非流式输出：一次性返回完整文本
        """
        messages = self._messages(system, user)
        stop = stop or ["</s>", "<|endoftext|>"]

        sig = inspect.signature(self.llm.create_chat_completion)
        kwargs = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        if "stop" in sig.parameters:
            kwargs["stop"] = stop

        out = self.llm.create_chat_completion(**kwargs)
        try:
            return (out["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return str(out).strip()