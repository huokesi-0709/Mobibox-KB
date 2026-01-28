from __future__ import annotations
import numpy as np
import sounddevice as sd

def record(seconds: float = 4.0, sample_rate: int = 16000) -> np.ndarray:
    """
    录音（mono, float32），返回 numpy 数组 shape=(n_samples,)
    """
    n = int(seconds * sample_rate)
    audio = sd.rec(n, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1)