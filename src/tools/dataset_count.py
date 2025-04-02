
from datasets import load_dataset

def main():
    # データセットの読み込み（splitは "train" と仮定）
    dataset = load_dataset("Atotti/VTuber-overview")
    if "train" in dataset:
        ds = dataset["train"]
    else:
        # 他のsplitが存在する場合、最初のものを使用
        ds = dataset[list(dataset.keys())[0]]

    # "markdown" カラムが存在するか確認
    if "markdown" not in ds.column_names:
        print("Error: Dataset に 'markdown' カラムが存在しません。")
        return

    # DataFrame に変換
    df = ds.to_pandas()
    # 欠損値がある場合は空文字に置換
    df["markdown"] = df["markdown"].fillna("").astype(str)
    # 各サンプルの文字数を計算
    df["char_count"] = df["markdown"].apply(len)

    # 統計情報を算出
    stats = df["char_count"].describe()

    print("=== 生成テキスト（markdown）の文字数に関する統計 ===")
    print(stats.to_string())

if __name__ == "__main__":
    main()
