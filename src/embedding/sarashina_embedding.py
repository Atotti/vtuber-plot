import os
import numpy as np
from typing import List
from src import vtuber, utils

def calc_embeddings():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sbintuitions/sarashina-embedding-v1-1b")

    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Loaded {len(vtubers_data)} vtubers!")

    names: List[str] = []
    sentences: List[str] = []

    for v in vtubers_data:
        name = utils.sanitize_path(v.name)
        save_path = f"data/sarashina_embedding/{name}.npy"
        if os.path.exists(save_path):
            print(f"⚠️  Embedding file already exists: {save_path}. Skipping.")
            continue

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
    embedding_vectors = model.encode(sentences, convert_to_numpy=True, batch_size=2)

    for name, embedding_vector in zip(names, embedding_vectors):
        save_path = f"data/sarashina_embedding/{name}.npy"
        np.save(save_path, embedding_vector)
        print(f"👌 Saved {save_path}")
