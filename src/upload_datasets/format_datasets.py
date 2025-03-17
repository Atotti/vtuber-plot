import os
import re
import json
import argparse
from datasets import Dataset

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

def sanitize_path(text: str) -> str:
    # にじさんじという文字を空白に置換し、ファイル名として使えない文字も除去
    sanitized = re.sub(r"にじさんじ", " ", text)
    sanitized = re.sub(r'[\\/:\*\?"<>|\-【】]', " ", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    return sanitized.strip()

def load_vtubers(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_dataset(json_path: str, markdown_dir: str = None) -> Dataset:
    vtubers = load_vtubers(json_path)
    names = []
    subscribers = []
    brands = []
    research_prompts = []
    markdown_contents = []

    for vtuber in vtubers:
        # 各 VTuber 情報を取得
        name = vtuber.get("name", "").strip()
        sub = vtuber.get("subscribers", None)
        brand_id = vtuber.get("brand_id", None)
        research_prompt = vtuber.get("research_prompt", "").strip()

        # ブランド名に変換（存在しなければ "Unknown"）
        brand_name = brand_id_dict.get(brand_id, "Unknown")

        names.append(name)
        subscribers.append(sub)
        brands.append(brand_name)
        research_prompts.append(research_prompt)

        # マークダウンファイルの存在チェック（存在すれば内容を取得）
        md_content = None
        if markdown_dir:
            sanitized_name = sanitize_path(name)
            md_path = os.path.join(markdown_dir, f"{sanitized_name}.md")
            if os.path.exists(md_path):
                try:
                    with open(md_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                except Exception as e:
                    print(f"Failed to read {md_path}: {e}")
        markdown_contents.append(md_content)

    data_dict = {
        "name": names,
        "subscribers": subscribers,
        "brand": brands,
        "research_prompt": research_prompts,
        "markdown": markdown_contents,
    }

    dataset = Dataset.from_dict(data_dict)
    return dataset

def main():
    parser = argparse.ArgumentParser(
        description="filtered_vtubers.json の情報（subscribers, brand, research_prompt）とマークダウンを Hugging Face Datasets としてアップロードするスクリプト"
    )
    parser.add_argument(
        "--json",
        type=str,
        default="data/filtered_vtubers.json",
        help="VTuber 情報の JSON ファイルのパス"
    )
    parser.add_argument(
        "--markdown_dir",
        type=str,
        default="data/SearchGPT",
        help="マークダウンファイルが保存されているディレクトリのパス"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="アップロード先の Hugging Face Hub リポジトリ名（例: username/vtuber-dataset）"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face のアクセストークン"
    )
    args = parser.parse_args()

    print("VTuber 情報の読み込み開始...")
    dataset = create_dataset(args.json, args.markdown_dir)
    print(f"データセット作成完了: {len(dataset)} 件のサンプルが読み込まれました。")

    print("データセットを Hugging Face Hub にアップロード中...")
    dataset.push_to_hub(args.repo, token=args.token)
    print("アップロード完了！")

if __name__ == "__main__":
    main()
