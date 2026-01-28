from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass
class WhisperASRConfig:
    model_dir: str                      # 本地目录：models/asr/faster-whisper-small
    device: str = "cpu"                 # cpu/cuda
    compute_type: str = "int8"          # int8/float16/float32
    language: str = "zh"


class FasterWhisperASR:
    def __init__(self, cfg: WhisperASRConfig):
        p = Path(cfg.model_dir).resolve()
        if not p.exists():
            raise FileNotFoundError(f"找不到 whisper 模型目录：{p}")
        self.cfg = cfg
        # 关键：传入本地目录
        self.model = WhisperModel(str(p), device=cfg.device, compute_type=cfg.compute_type)

    def transcribe(self, audio) -> str:
        segments, info = self.model.transcribe(audio, language=self.cfg.language, vad_filter=True)
        return "".join([seg.text for seg in segments]).strip()