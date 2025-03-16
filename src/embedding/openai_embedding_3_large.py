import os
import dotenv
import numpy as np
from typing import List
# 新しい OpenAI Python ライブラリ を使用
# pip install --upgrade openai
from openai import OpenAI

from src import vtuber, utils

def calc_embeddings():
    # .env ファイルから環境変数を読み込む（OPENAI_API_KEY など）
    dotenv.load_dotenv()
    # OpenAI クライアントの初期化
    client = OpenAI()

    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Loaded {len(vtubers_data)} vtubers!")

    names: List[str] = []
    sentences: List[str] = []

    # 埋め込みの保存先フォルダ
    output_dir = "data/text-embedding-3-large"
    os.makedirs(output_dir, exist_ok=True)

    for v in vtubers_data:
        name = utils.sanitize_path(v.name)
        save_path = os.path.join(output_dir, f"{name}.npy")

        # 既に埋め込みファイルがある場合はスキップ
        if os.path.exists(save_path):
            print(f"⚠️  Embedding file already exists: {save_path}. Skipping.")
            continue

        # 対応する .md ファイルを読み込む
        md_path = f"data/SearchGPT/{name}.md"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"⚠️  File not found: {md_path}")
            continue

        names.append(name)
        sentences.append(content)

    if not sentences:
        print("No new embeddings to calculate. Exiting.")
        return

    print("🫠 Start calculating embeddings...")

    batch_size = 4

    for start_idx in range(0, len(sentences), batch_size):
        batch_sentences = sentences[start_idx : start_idx + batch_size]
        batch_names = names[start_idx : start_idx + batch_size]

        # OpenAI Embeddings API による埋め込み取得
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch_sentences
        )

        # response はオブジェクトなので、response.data から取り出す
        for i, name in enumerate(batch_names):
            embedding_vector = response.data[i].embedding  # ここがポイント
            save_path = os.path.join(output_dir, f"{name}.npy")
            np.save(save_path, embedding_vector)
            print(f"👌 Saved {save_path}")
