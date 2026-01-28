from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

from llama_cpp import Llama
from monibox_kb.paths import PROJECT_ROOT


@dataclass
class LLMConfig:
    gguf_path: str
    n_ctx: int = 2048
    n_threads: int = 6
    n_gpu_layers: int = 0


class LlamaCppChat:
    """
    llama-cpp-python 封装：支持 generate 与 stream。
    """
    def __init__(self, cfg: LLMConfig):
        p = Path(cfg.gguf_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到 GGUF：{p}")

        self.llm = Llama(
            model_path=str(p),
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )

    def stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.6, top_p: float = 0.9) -> Iterator[str]:
        for chunk in self.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                              stop=["</s>", "<|endoftext|>"], stream=True):
            tok = chunk["choices"][0].get("text", "")
            if tok:
                yield tok

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.6, top_p: float = 0.9) -> str:
        out = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=["</s>", "<|endoftext|>"])
        return out["choices"][0]["text"].strip()


