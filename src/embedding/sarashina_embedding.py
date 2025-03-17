import os
import numpy as np
from typing import List
from datasets import load_dataset
from src import vtuber, utils


def calc_embeddings(dataset_name:str = "Atotti/VTuber-overview", split:str = "train"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sbintuitions/sarashina-embedding-v1-1b")

    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Loaded {len(vtubers_data)} vtubers!")

    dataset = load_dataset(dataset_name, split=split)

    if "markdown" not in dataset.column_names:
        if "text" in dataset.column_names:
            dataset = dataset.rename_column("text", "markdown")
        else:
            print("⚠️ Dataset has neither 'markdown' nor 'text' column. Exiting.")
            return

    md_map = {}
    for row in dataset:
        md_map[row["name"]] = row["markdown"]

    names: List[str] = []
    sentences: List[str] = []

    for v in vtubers_data:
        name = utils.sanitize_path(v.name)
        save_path = f"data/sarashina_embedding/{name}.npy"
        if os.path.exists(save_path):
            print(f"⚠️  Embedding file already exists: {save_path}. Skipping.")
            continue

        content = md_map.get(name, "")
        if not content:
            print(f"⚠️  No markdown found in dataset for: {name}")
            continue

        names.append(name)
        sentences.append(content)

    if not sentences:
        print("No new embeddings to calculate. Exiting.")
        return

    print("🫠 Start calculating embeddings...")
    embedding_vectors = model.encode(sentences, convert_to_numpy=True, batch_size=2)

    for name, embedding_vector in zip(names, embedding_vectors):
        save_path = f"data/sarashina_embedding/{split}/{name}.npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embedding_vector)
        print(f"👌 Saved {save_path}")
