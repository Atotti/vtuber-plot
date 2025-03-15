import os
from openai import OpenAI
import dotenv

class SearchGPT():
    def __init__(self):
        dotenv.load_dotenv()
        self.MODEL = os.getenv("MODEL", "gpt-4o-search-preview")
        self.client = OpenAI()

    def send(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.MODEL,
            web_search_options={
                "user_location": {
                    "type": "approximate",
                    "approximate": {
                        "country": "JP",
                    },
                "search_context_size": "high",
                },
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return completion.choices[0].message.content


if __name__=="__main__":
    sample_prompt="しぐれういというVTuberについて以下の点に重視して詳細に調査し長文で解説してください。\n- キャラクター性\n- 活動内容\n- コラボ履歴\n- 人間関係\n- コンテンツのジャンル\n- 活動スタイル\n- 他のVTuberと比較した時の特徴\n"
    gpt = SearchGPT()
    response = gpt.send(sample_prompt)

