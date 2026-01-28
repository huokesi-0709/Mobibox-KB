from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

from llama_cpp import Llama
from monibox_kb.paths import PROJECT_ROOT

@dataclass
class LLMConfig:
    gguf_path: str
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0

class LlamaCppChat:
    """
    极简封装：输入 prompt -> 输出文本
    说明：不同 GGUF/模型可能需要不同 chat_format。这里先用稳健的“纯prompt”方式跑通。
    你后续想切到 create_chat_completion / chat_format，我也可以再帮你适配 Qwen 的格式。
    """
    def __init__(self, cfg: LLMConfig):
        path = cfg.gguf_path
        if not os.path.isabs(path):
            path = str((PROJECT_ROOT / path).resolve())

        self.llm = Llama(
            model_path=path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads,
            n_gpu_layers=cfg.n_gpu_layers,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.6, top_p: float = 0.9) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["</s>", "<|endoftext|>"]
        )
        return out["choices"][0]["text"].strip()