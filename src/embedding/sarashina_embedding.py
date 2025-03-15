from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

from src import vtuber, utils


def calc_embeddings():
    model = SentenceTransformer("sbintuitions/sarashina-embedding-v1-1b")

    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Loaded {len(vtubers_data)} vtubers!")

    names: List[str] = []
    sentences: List[str] = []

    for v in vtubers_data:
        name = utils.sanitize_path(v.name)
        md_path = f"data/SearchGPT/{name}.md"
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"⚠️  File not found: {md_path}")
            continue

        names.append(name)
        sentences.append(content)

    print("🫠 Start calculating embeddings...")

    embedding_vectors = model.encode(sentences, convert_to_numpy=True, batch_size=2)

    for name, embedding_vector in zip(names, embedding_vectors):
        save_path = f"data/sarashina_embedding/{name}.npy"
        np.save(save_path, embedding_vector)
        print(f"👌 Saved {save_path}")
