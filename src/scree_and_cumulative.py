import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def plot_pca_scree_and_cumulative(X, embedding_dir):
    """
    与えられたデータ X に PCA を適用し、
    Scree Plot と Cumulative Explained Variance Plot を描画する。
    """
    # PCA を適用
    pca = PCA()  # デフォルトでは成分数は特徴量数まで自動的に考慮
    pca.fit(X)

    # 各主成分の寄与率
    explained_variances = pca.explained_variance_ratio_
    n_components = len(explained_variances)
    x_labels = range(1, n_components + 1)  # x 軸 (1, 2, 3, ...)

    # 1. Scree Plot: 各主成分の寄与率を棒グラフで表示
    plt.figure()
    plt.bar(x_labels[:25], explained_variances[:25])
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")
    plt.savefig(embedding_dir + "/scree_plot.png")

    # 2. Cumulative Explained Variance Plot: 寄与率の累積和を折れ線グラフで表示
    plt.figure()
    plt.plot(x_labels, np.cumsum(explained_variances), marker='o')
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.savefig(embedding_dir + "/cumulative_explained_variance.png")

embedding_dir="data/text-embedding-3-large"

# 埋め込みファイルの読み込み
embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".npy")]
names, embeddings, brands = [], [], []
for file in embedding_files:
    path = os.path.join(embedding_dir, file)
    emb = np.load(path)
    sanitized_name = os.path.splitext(file)[0]

    names.append(sanitized_name)
    embeddings.append(emb)

# PCAにかけるため、(サンプル数, 埋め込み次元数)の形に変形
X = np.vstack(embeddings)

plot_pca_scree_and_cumulative(X, embedding_dir)
