"""
scripts/test_llm_chat.py

用途
-----
最小可运行测试：验证 llama-cpp-python + Qwen1.5 GGUF 是否可用，并验证：
- Chat streaming 是否正常产出 token
- 强制 JSON 输出是否可被 utils_json.extract_first_json 稳定解析（支持多 JSON 连续输出）

运行方式
--------
python -m scripts.test_llm_chat
或 PyCharm 直接运行（本脚本会自行加载项目根目录 .env）。
"""

import os
from dotenv import load_dotenv

from monibox_kb.paths import PROJECT_ROOT
from monibox_kb.llm.llama_cpp_chat import LLMConfig, LlamaCppChat
from monibox_kb.utils_json import extract_first_json


def main():
    # 1) 加载 .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        print(f"[WARN] 未找到 .env：{env_path}")

    llm_path = (os.getenv("LLM_GGUF_PATH") or "").strip()
    if not llm_path:
        raise RuntimeError(
            "未读取到环境变量 LLM_GGUF_PATH。\n"
            f"已尝试加载 .env：{env_path}\n"
        )

    cfg = LLMConfig(
        gguf_path=llm_path,
        n_ctx=int(os.getenv("LLM_CTX", "2048")),
        n_threads=int(os.getenv("LLM_THREADS", "6")),
        n_gpu_layers=int(os.getenv("LLM_GPU_LAYERS", "0")),
    )
    llm = LlamaCppChat(cfg)

    # system：强调“只输出一个 JSON 并立即停止”
    system = (
        "你是灾害受困陪伴设备 MoniBox。"
        "你必须只输出【一个】JSON对象，输出后立刻停止，不要再输出第二个JSON，不要解释。"
        "JSON格式必须包含3个key：text, used_ids, ask。"
        '示例：{"text":"先慢慢吸气4拍呼气6拍。","used_ids":["k_demo_1"],"ask":""}'
    )

    user = (
        '已检索要点：\n'
        '- id="k_demo_1" text="先用4拍吸气、6拍呼气，重复3轮，帮助稳定呼吸。"\n'
        '- id="k_demo_2" text="如果有粉尘，尽量用衣物遮住口鼻，减少说话。"\n\n'
        "用户：我好害怕，喘不过气\n"
    )

    print("[TEST] PROJECT_ROOT =", PROJECT_ROOT)
    print("[TEST] .env path     =", env_path)
    print("[TEST] chat_format   =", llm.chat_format)
    print("[TEST] model_path    =", llm_path)
    print("\n[TEST] streaming output:\n")

    buf = ""
    # stop 关键：阻止模型输出第二个 JSON（通常以换行+{ 开始）
    stop = ["\n{", "\r\n{", "</s>", "<|endoftext|>"]

    for tok in llm.stream_chat(system, user, max_tokens=220, temperature=0.3, top_p=0.9, stop=stop):
        buf += tok
        print(tok, end="", flush=True)

    print("\n\n[TEST] raw output:\n", buf)

    obj = extract_first_json(buf)
    print("\n[TEST] parsed json:\n", obj)


if __name__ == "__main__":
    main()