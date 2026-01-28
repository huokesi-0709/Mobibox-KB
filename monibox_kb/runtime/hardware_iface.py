from __future__ import annotations
from typing import Any, Dict

class HardwareIface:
    """真实硬件接口：后续在 Radxa 上实现"""
    def tts(self, text: str, style: str | None = None): ...
    def led(self, pattern: Dict[str, Any]): ...
    def screen(self, text: str, ms: int = 2000): ...

class MockHardware(HardwareIface):
    """PC 上调试用：只打印"""
    def tts(self, text: str, style: str | None = None):
        print(f"[TTS style={style}] {text}")

    def led(self, pattern: Dict[str, Any]):
        print(f"[LED] {pattern}")

    def screen(self, text: str, ms: int = 2000):
        print(f"[SCREEN {ms}ms] {text}")