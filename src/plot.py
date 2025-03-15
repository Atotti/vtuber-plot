import os
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA


def plot_embeddings_with_pca(embedding_dir="data/sarashina_embedding"):
    embedding_files = [
        f for f in os.listdir(embedding_dir) if f.endswith(".npy")
    ]

    # 2. 埋め込みとファイル名を読み込む
    names = []
    embeddings = []
    for file in embedding_files:
        path = os.path.join(embedding_dir, file)
        emb = np.load(path)
        embeddings.append(emb)
        # 拡張子を除いた部分を名前とする
        names.append(os.path.splitext(file)[0])

    # 3. 埋め込みを numpy 配列にまとめる
    #    形状: (サンプル数, 埋め込み次元数)
    X = np.vstack(embeddings)

    # 4. PCA で 2 次元に圧縮
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 5. Matplotlibで散布図をプロット
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)

    # 6. 点の近くにラベル(名前)を付与
    for i, name in enumerate(names):
        plt.annotate(name, (X_2d[i, 0], X_2d[i, 1]))

    plt.title("VTuber Embeddings PCA Scatter Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig("plot-sarashina-embedding.pdf")
