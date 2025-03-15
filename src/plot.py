import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from adjustText import adjust_text
import pandas as pd
import japanize_matplotlib

from src import vtuber, utils

brand_id_dict = {
    1: "個人",
    2: "ホロライブ",
    3: "ホロスターズ",
    20: "ぶいすぽ",
    18: "エイレーン",
    17: "のりプロ",
    92: "あおぎり高校",
    89: "ななしいんく",
    162: "ネオポルテ",
    7: "にじさんじ",
    31: "Kizuna AI"
}

def plot_embeddings_with_pca(
    embedding_dir="data/sarashina_embedding",
    vtubers_json_path="data/filtered_vtubers.json"
):

    # 1. VTuber一覧を読み込んで、名前(正規化済) -> ブランドID の辞書を作る
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)

    name_to_brand = {}
    for v in vtubers_data:
        sanitized_name = utils.sanitize_path(v.name)
        brand = v.brand_id if v.brand_id else "Unknown"
        name_to_brand[sanitized_name] = brand

    # 2. 埋め込みファイルの読み込み
    embedding_files = [
        f for f in os.listdir(embedding_dir) if f.endswith(".npy")
    ]

    names = []
    embeddings = []
    brands = []

    for file in embedding_files:
        path = os.path.join(embedding_dir, file)
        emb = np.load(path)

        sanitized_name = os.path.splitext(file)[0]
        brand = name_to_brand.get(sanitized_name, "Unknown")

        names.append(sanitized_name)
        embeddings.append(emb)
        brands.append(brand_id_dict[brand])

    # PCAにかけるため、(サンプル数, 埋め込み次元数)の形に変形
    X = np.vstack(embeddings)

    # 3. PCAで2次元に圧縮
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 可視化用に DataFrame を作成
    df = pd.DataFrame({
        "name": names,
        "brand_id": brands,
        "x": X_2d[:, 0],
        "y": X_2d[:, 1],
    })

    # 4. Seabornで散布図を作成
    plt.figure(figsize=(10, 8), dpi=500)
    sns.set_theme(style="ticks", context="notebook", font="IPAexGothic")

    scatter = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="brand_id",
        palette="Set1",
        alpha=0.8,
        edgecolor="none",
        legend=False
    )

    plt.title("VTuber プロット (sarashina embedding)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # 5. テキストラベルを重ならないようにadjustTextで調整
    texts = []
    for i, row in df.iterrows():
        texts.append(plt.text(row["x"], row["y"], row["name"], fontsize=8, color="gray"))

    # ラベルを自動的に調整
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5)
    )

    plt.tight_layout()
    plt.savefig("plot-sarashina-embedding.png")
    plt.savefig("plot-sarashina-embedding.pdf")
