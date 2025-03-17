import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from src import vtuber, utils

HORIZONTAL_AXIS = 2
VERTICAL_AXIS = 3

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


def plot_embeddings_interactive(
    embedding_dir="data/sarashina_embedding",
    vtubers_json_path="data/filtered_vtubers.json",
):
    embedding_model = embedding_dir.split("/")[-1]
    # 1. VTuber一覧を読み込んで、名前(正規化済) -> ブランドID の辞書を作成
    vtubers_data = vtuber.load_vtubers(vtubers_json_path)
    name_to_brand = {}
    for v in vtubers_data:
        sanitized_name = utils.sanitize_path(v.name)
        brand = v.brand_id if v.brand_id else "Unknown"
        name_to_brand[sanitized_name] = brand

    # 2. 埋め込みファイルの読み込み
    embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".npy")]
    names, embeddings, brands = [], [], []
    for file in embedding_files:
        path = os.path.join(embedding_dir, file)
        emb = np.load(path)
        sanitized_name = os.path.splitext(file)[0]
        brand = name_to_brand.get(sanitized_name, "Unknown")
        # brand_id_dictで変換
        brands.append(brand_id_dict[brand])
        names.append(sanitized_name)
        embeddings.append(emb)

    # 3. PCAで2次元に圧縮
    X = np.vstack(embeddings)
    pca = PCA(n_components=max(VERTICAL_AXIS, HORIZONTAL_AXIS))
    X_2d = pca.fit_transform(X)

    # 4. DataFrame作成
    df = pd.DataFrame(
        {
            "name": names,
            "brand_id": brands,
            f"PC{str(HORIZONTAL_AXIS)}": X_2d[:, HORIZONTAL_AXIS-1],
            f"PC{str(VERTICAL_AXIS)}": X_2d[:, VERTICAL_AXIS-1],
        }
    )

    # 5. Plotly Expressでインタラクティブな散布図を作成
    fig = px.scatter(
        df,
        x=f"PC{str(HORIZONTAL_AXIS)}",
        y=f"PC{str(VERTICAL_AXIS)}",
        color="brand_id",
        color_discrete_map=color_map,
        hover_name="name",  # ホバー時に名前だけ表示
        hover_data={},  # その他の情報は非表示
        title=f"VTuber プロット ({embedding_model})",
    )

    # 点のサイズを大きくする（例として15に設定）
    fig.update_traces(marker=dict(size=15))

    # HTMLファイルとして保存（Webに埋め込む際などに利用）
    fig.write_html(f"plot-{embedding_model}.html")
    fig.write_html(f"plot-{embedding_model}-{str(HORIZONTAL_AXIS)}-{str(VERTICAL_AXIS)}.html")

