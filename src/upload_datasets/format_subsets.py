import os
import re
import json
import argparse
import traceback
from datasets import Dataset, DatasetDict

# ======= 設定例: ブランドIDに対するブランド名辞書 =======
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


# ======= ファイル名に利用できない文字などを除去するユーティリティ関数 =======
def sanitize_path(text: str) -> str:
    """にじさんじという文字を空白に置換し、ファイル名として使えない文字も除去"""
    text = re.sub(r"にじさんじ", " ", text)
    text = re.sub(r'[\\/:\*\?"<>|\-【】]', " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ======= filtered_vtubers.json の読み込み =======
def load_vtubers(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ======= Markdownから各セクションをパースする関数 =======
def parse_markdown_sections(md_content: str) -> dict:
    """
    md_content 内にある以下のセクションを抜き出す:
      - キャラクター性
      - 活動内容
      - コラボ履歴
      - 人間関係
      - コンテンツのジャンル
      - 活動スタイル
      - 他のVTuberと比較した時の特徴

    見つからなかった場合は空文字を返す。
    """
    sections = {
        "キャラクター性": "",
        "活動内容": "",
        "コラボ履歴": "",
        "人間関係": "",
        "コンテンツのジャンル": "",
        "活動スタイル": "",
        "他のVTuberと比較した時の特徴": "",
    }

    for section_title in sections.keys():
        # "**キャラクター性**" といったマークダウン見出しから
        # 次の見出し or ファイル終端 までを正規表現で抜き出す
        pattern = rf"\*\*{section_title}\*\*\s*\n([\s\S]*?)(?=\n\*\*|$)"
        match = re.search(pattern, md_content)
        if match:
            sections[section_title] = match.group(1).strip()

    return sections


def remove_markdown_links(text: str) -> str:
    """
    Markdownリンク記法を削除する。
    例: [テキスト](URL) や ([テキスト](URL)) といったパターンを削除

    - パターン1: \(\[[^\]]+\]\([^)]*\)\)
      -> ([text](link)) のように丸括弧で囲まれているリンク
    - パターン2: \[[^\]]+\]\([^)]*\)
      -> [text](link) の基本的なMarkdownリンク
    """
    # 1. 丸括弧ごと ([text](link)) を削除
    text = re.sub(r"\(\[[^\]]+\]\([^)]*\)\)", "", text)

    # 2. 通常の [text](link) だけを削除
    text = re.sub(r"\[[^\]]+\]\([^)]*\)", "", text)

    # 連続した空白が生じた場合などを整形
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(
        description="MarkdownファイルからVTuber情報をパースし、リンクを除去した上で各セクションごとにDatasetのサブセットを作成してHugging Face Hubへアップロードするスクリプト"
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
        help="Markdownファイルが保存されているディレクトリのパス"
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
    vtubers = load_vtubers(args.json)
    print(f"{len(vtubers)} 件のVTuber情報を読み込みました。")

    # 各セクションごとに格納するリスト（DatasetDictの各split用）
    sections_dict = {
        "キャラクター性": {"name": [], "brand": [], "subscribers": [], "text": []},
        "活動内容": {"name": [], "brand": [], "subscribers": [], "text": []},
        "コラボ履歴": {"name": [], "brand": [], "subscribers": [], "text": []},
        "人間関係": {"name": [], "brand": [], "subscribers": [], "text": []},
        "コンテンツのジャンル": {"name": [], "brand": [], "subscribers": [], "text": []},
        "活動スタイル": {"name": [], "brand": [], "subscribers": [], "text": []},
        "他のVTuberと比較した時の特徴": {"name": [], "brand": [], "subscribers": [], "text": []},
    }

    # ======= マークダウンファイルを読み込み、該当セクションを抽出 =======
    for vtuber in vtubers:
        name = vtuber.get("name", "").strip()
        sub = vtuber.get("subscribers", None)
        brand_id = vtuber.get("brand_id", None)
        # brand_id_dict からブランド名を取得 (なければ "Unknown")
        brand_name = brand_id_dict.get(brand_id, "Unknown")

        # 対応するマークダウンファイルを探す
        sanitized_name = sanitize_path(name)
        md_path = os.path.join(args.markdown_dir, f"{sanitized_name}.md")

        if not os.path.exists(md_path):
            # 存在しない場合は空のまま追加
            for section_title in sections_dict.keys():
                sections_dict[section_title]["name"].append(name)
                sections_dict[section_title]["brand"].append(brand_name)
                sections_dict[section_title]["subscribers"].append(sub)
                sections_dict[section_title]["text"].append("")
            continue

        # ファイルを読み込んでパース
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            parsed_sections = parse_markdown_sections(md_content)

            # パースしたテキストからMarkdownリンクを除去
            for section_title, text in parsed_sections.items():
                cleaned_text = remove_markdown_links(text)
                sections_dict[section_title]["name"].append(name)
                sections_dict[section_title]["brand"].append(brand_name)
                sections_dict[section_title]["subscribers"].append(sub)
                sections_dict[section_title]["text"].append(cleaned_text)

        except Exception as e:
            traceback.print_exc()
            print(f"Failed to read or parse {md_path} for {name}: {e}")
            # エラーがあった場合も、空のままレコードを追加
            for section_title in sections_dict.keys():
                sections_dict[section_title]["name"].append(name)
                sections_dict[section_title]["brand"].append(brand_name)
                sections_dict[section_title]["subscribers"].append(sub)
                sections_dict[section_title]["text"].append("")

    # ======= DatasetDict の作成（各セクションをそれぞれの split として格納） =======
    print("DatasetDict を作成します...")
    dataset_splits = {}
    for section_title, data in sections_dict.items():
        dataset_splits[section_title] = Dataset.from_dict(data)

    dataset_dict = DatasetDict(dataset_splits)
    print("セクションごとに Dataset を作成しました。")

    # ======= Hugging Face Hub へアップロード =======
    print(f"Hugging Face Hubへアップロードします -> repo: {args.repo}")
    dataset_dict.push_to_hub(args.repo, token=args.token)
    print("アップロード完了！")


if __name__ == "__main__":
    main()
