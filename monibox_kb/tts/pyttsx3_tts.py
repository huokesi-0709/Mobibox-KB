from __future__ import annotations
import pyttsx3

class Pyttsx3TTS:
    def __init__(self, rate: int = 180, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def speak(self, text: str):
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()