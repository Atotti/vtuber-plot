import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from adjustText import adjust_text
import pandas as pd
import japanize_matplotlib
from src import vtuber, utils, label_v

HORIZONTAL_AXIS = 2
VERTICAL_AXIS = 3

# brand_id_dict を使ってブランド名を定義
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
    31: "Kizuna AI",
    57: "深層組",
    53: "Crazy Raccoon",
    127: "REJECT",
}

# カスタムカラーマッピング
color_map = {
    "個人": "gray",
    "ホロライブ": "#ff80bf",
    "ホロスターズ": "#1e90ff",
    "ぶいすぽ": "#ff4500",
    "エイレーン": "#00ced1",
    "のりプロ": "#32cd32",
    "あおぎり高校": "#000080",
    "ななしいんく": "#ffd700",
    "ネオポルテ": "#8a2be2",
    "にじさんじ": "#800080",
    "Kizuna AI": "#ff1493",
    "深層組": "#191970",
    "Crazy Raccoon": "#ff8c00",
    "REJECT": "#696969",
}


def plot_embeddings_with_tsne(
    embedding_dir="data/sarashina_embedding",
    vtubers_json_path="data/filtered_vtubers.json",
    HORIZONTAL_AXIS=HORIZONTAL_AXIS,
    VERTICAL_AXIS=VERTICAL_AXIS
):
    embedding_model = "/".join(embedding_dir.split("/")[1:])

    # 1. VTuber一覧を読み込んで、名前（正規化済）→ ブランドID の辞書を作成
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)
    name_to_brand = {}
    text_label = set()
    for v in vtubers_data:
        sanitized_name = utils.sanitize_path(v.name)
        # ブランドIDがなければ "Unknown" とする
        brand = v.brand_id if v.brand_id else "Unknown"
        name_to_brand[sanitized_name] = brand
        if v.name in label_v.label_v:
            text_label.add(sanitized_name)

    # 2. 埋め込みファイルの読み込み
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".npy")]
    names, embeddings, brands = [], [], []
    for file in embedding_files:
        path = os.path.join(embedding_dir, file)
        emb = np.load(path)
        sanitized_name = os.path.splitext(file)[0]
        brand = name_to_brand.get(sanitized_name, "Unknown")
        names.append(sanitized_name)
        embeddings.append(emb)
        # brand_id_dict に従ってブランド名に変換
        brands.append(brand_id_dict[brand])

    # 3. PCAにかけるため、(サンプル数, 埋め込み次元数)の形に変形
    X = np.vstack(embeddings)

    # 4. PCAで2次元に圧縮
    pca = TSNE(n_components=max(VERTICAL_AXIS, HORIZONTAL_AXIS))
    X_2d = pca.fit_transform(X)

    # 5. 可視化用 DataFrame の作成
    df = pd.DataFrame(
        {
            "name": names,
            "brand_id": brands,
            "x": X_2d[:, HORIZONTAL_AXIS-1],
            "y": X_2d[:, VERTICAL_AXIS-1],
        }
    )

    # 6. Seaborn の設定（日本語表示のために IPAexGothic を指定）
    plt.figure(figsize=(16, 10), dpi=800)
    sns.set_theme(style="ticks", context="notebook", font="IPAexGothic")

    # 7. 散布図作成
    scatter = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="brand_id",
        palette=color_map,  # カスタムカラーマッピングを利用
        s=80,
        alpha=0.8,
        edgecolor="none",
        legend="full",  # 凡例を表示
    )

    plt.title(f"VTuber プロット ({embedding_model})")
    plt.xlabel(f"PC{str(HORIZONTAL_AXIS)}")
    plt.ylabel(f"PC{str(VERTICAL_AXIS)}")

    plt.legend(
        title="ブランド",
        bbox_to_anchor=(1.05, 1),  # 右側の外に配置
        loc="upper left",  # 左上寄せ
        borderaxespad=0,
    )

    # 8. テキストラベルを配置（重なり防止のため adjustText を使用）
    texts = []
    for i, row in df.iterrows():
        if row["name"] in text_label:
            texts.append(plt.text(row["x"], row["y"], row["name"], fontsize=12))

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    plt.tight_layout()
    os.makedirs(f"works/TSNE/{embedding_model}/", exist_ok=True)
    plt.savefig(f"works/TSNE/{embedding_model}/{str(HORIZONTAL_AXIS)}-{str(VERTICAL_AXIS)}.png")
    plt.savefig(f"works/TSNE/{embedding_model}/{str(HORIZONTAL_AXIS)}-{str(VERTICAL_AXIS)}.pdf")
