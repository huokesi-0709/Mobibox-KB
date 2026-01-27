"""
DeepSeek API 封装（OpenAI 兼容）。
你现在用 deepseek-chat；后面切换 reasoner 只改调用参数。
"""
from openai import OpenAI
from monibox_kb.config import settings

class DeepSeekClient:
    def __init__(self):
        if not settings.deepseek_api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在 .env 中配置")
        self.client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url
        )

    def chat(self, system: str, user: str, temperature: float = 0.7) -> str:
        resp = self.client.chat.completions.create(
            model=settings.deepseek_model_chat,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature
        )
        return resp.choices[0].message.content

    def reasoner(self, system: str, user: str, temperature: float = 0.3) -> str:
        """
        预留：当你要生成“体系/变量矩阵/复杂约束”时用 reasoner。
        """
        resp = self.client.chat.completions.create(
            model=settings.deepseek_model_reasoner,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature
        )
        return resp.choices[0].message.content