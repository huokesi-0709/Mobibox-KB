import os
import argparse

from monibox_kb.config import settings
from monibox_kb.audio.recorder import record
from monibox_kb.asr.faster_whisper_asr import FasterWhisperASR, WhisperASRConfig
from monibox_kb.runtime.session import MoniSession, SessionConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mic", "text"], default="mic")
    ap.add_argument("--text", default="")
    ap.add_argument("--events", default="", help="逗号分隔，如 imu_strong_shake")
    ap.add_argument("--auto_top_tags", type=int, default=2)
    ap.add_argument("--no_tts", action="store_true", help="只在控制台输出，不播放语音")
    args = ap.parse_args()

    events = [e.strip() for e in args.events.split(",") if e.strip()]

    # --- ASR: faster-whisper from local dir ---
    asr_cfg = WhisperASRConfig(
        model_dir=os.getenv("WHISPER_MODEL_DIR", "models/asr/faster-whisper-small"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        language=os.getenv("WHISPER_LANGUAGE", "zh"),
    )
    asr = FasterWhisperASR(asr_cfg)

    # --- Session: Protocol + RAG + LLM(stream) + Safety + TTS ---
    llm_path = os.getenv("LLM_GGUF_PATH", "")
    if not llm_path:
        raise RuntimeError("请在 .env 中设置 LLM_GGUF_PATH")

    sess_cfg = SessionConfig(
        llm_path=llm_path,
        llm_ctx=int(os.getenv("LLM_CTX", "2048")),
        llm_threads=int(os.getenv("LLM_THREADS", "6")),
        llm_gpu_layers=int(os.getenv("LLM_GPU_LAYERS", "0")),
        tts_enabled=(not args.no_tts),
    )
    session = MoniSession(settings.rag_db_path, sess_cfg)

    if args.mode == "text":
        user_text = args.text.strip()
        if not user_text:
            raise RuntimeError("--mode text 时必须提供 --text")
        print("[TEXT INPUT]", user_text)
        session.handle(user_text, events=events, auto_top_tags=args.auto_top_tags)
        return

    # mic mode
    sec = float(os.getenv("REC_SECONDS", "4"))
    sr = int(os.getenv("REC_SAMPLE_RATE", "16000"))

    print(f"按回车开始录音 {sec}s ...")
    input()
    audio = record(seconds=sec, sample_rate=sr)
    print("识别中...")
    user_text = asr.transcribe(audio)
    print("[ASR]", user_text)

    if not user_text:
        print("未识别到内容")
        return

    session.handle(user_text, events=events, auto_top_tags=args.auto_top_tags)


if __name__ == "__main__":
    main()