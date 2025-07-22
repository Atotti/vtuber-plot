# VTuber 埋め込みベクトル可視化プロジェクト

## 🎭 プロジェクト概要

このプロジェクトは、日本のVTuber（バーチャルYouTuber）の特徴を言語モデルを用いて埋め込みベクトル化し、その関係性を視覚的に探索できるようにした研究プロジェクトです。

VTuberの活動内容、キャラクター性、コラボレーション履歴などの多面的な特徴を分析し、2次元空間上にマッピングすることで、VTuber同士の類似性や関係性を直感的に理解できます。

## 📊 可視化結果を見る

### 🌟 おすすめの可視化

初めての方は、以下の可視化から始めることをおすすめします：

1. **[全カテゴリ統合版](works/sarashina_embedding/1-2.html)**
   - すべての特徴を統合した総合的な可視化
   - VTuber間の全体的な関係性が一目でわかります

2. **[活動内容別 - インタラクティブプロット](works/活動内容/sarashina_embedding/1-2.html)**
   - ゲーム実況、歌、雑談など活動内容による分類
   - マウスホバーで詳細情報を表示

3. **[人間関係別 - インタラクティブプロット](works/人間関係/sarashina_embedding/1-2.html)**
   - コラボレーションやグループ関係による分類
   - VTuberコミュニティの構造を視覚化

### 📁 すべての可視化結果

#### インタラクティブプロット（詳細版）

各カテゴリごとに、埋め込みベクトルの異なる次元の組み合わせを可視化しています：

<details>
<summary>🎨 キャラクター性</summary>

- [次元1-2](works/キャラクター性/sarashina_embedding/1-2.html)
- [次元1-3](works/キャラクター性/sarashina_embedding/1-3.html)
- [次元1-4](works/キャラクター性/sarashina_embedding/1-4.html)
- [次元1-5](works/キャラクター性/sarashina_embedding/1-5.html)
- [次元2-3](works/キャラクター性/sarashina_embedding/2-3.html)

</details>

<details>
<summary>🤝 コラボ履歴</summary>

- [次元1-2](works/コラボ履歴/sarashina_embedding/1-2.html)
- [次元1-3](works/コラボ履歴/sarashina_embedding/1-3.html)
- [次元1-4](works/コラボ履歴/sarashina_embedding/1-4.html)
- [次元1-5](works/コラボ履歴/sarashina_embedding/1-5.html)
- [次元2-3](works/コラボ履歴/sarashina_embedding/2-3.html)

</details>

<details>
<summary>🎮 コンテンツのジャンル</summary>

- [次元1-2](works/コンテンツのジャンル/sarashina_embedding/1-2.html)
- [次元1-3](works/コンテンツのジャンル/sarashina_embedding/1-3.html)
- [次元1-4](works/コンテンツのジャンル/sarashina_embedding/1-4.html)
- [次元1-5](works/コンテンツのジャンル/sarashina_embedding/1-5.html)
- [次元2-3](works/コンテンツのジャンル/sarashina_embedding/2-3.html)

</details>

<details>
<summary>👥 人間関係</summary>

- [次元1-2](works/人間関係/sarashina_embedding/1-2.html)
- [次元1-3](works/人間関係/sarashina_embedding/1-3.html)
- [次元1-4](works/人間関係/sarashina_embedding/1-4.html)
- [次元1-5](works/人間関係/sarashina_embedding/1-5.html)
- [次元2-3](works/人間関係/sarashina_embedding/2-3.html)

</details>

<details>
<summary>✨ 他のVTuberと比較した時の特徴</summary>

- [次元1-2](works/他のVTuberと比較した時の特徴/sarashina_embedding/1-2.html)
- [次元1-3](works/他のVTuberと比較した時の特徴/sarashina_embedding/1-3.html)
- [次元1-4](works/他のVTuberと比較した時の特徴/sarashina_embedding/1-4.html)
- [次元1-5](works/他のVTuberと比較した時の特徴/sarashina_embedding/1-5.html)
- [次元2-3](works/他のVTuberと比較した時の特徴/sarashina_embedding/2-3.html)

</details>

<details>
<summary>📺 活動スタイル</summary>

- [次元1-2](works/活動スタイル/sarashina_embedding/1-2.html)
- [次元1-3](works/活動スタイル/sarashina_embedding/1-3.html)
- [次元1-4](works/活動スタイル/sarashina_embedding/1-4.html)
- [次元1-5](works/活動スタイル/sarashina_embedding/1-5.html)
- [次元2-3](works/活動スタイル/sarashina_embedding/2-3.html)

</details>

<details>
<summary>🎯 活動内容</summary>

- [次元1-2](works/活動内容/sarashina_embedding/1-2.html)
- [次元1-3](works/活動内容/sarashina_embedding/1-3.html)
- [次元1-4](works/活動内容/sarashina_embedding/1-4.html)
- [次元1-5](works/活動内容/sarashina_embedding/1-5.html)
- [次元2-3](works/活動内容/sarashina_embedding/2-3.html)

</details>

<details>
<summary>🔍 全体（sarashina_embedding）</summary>

- [次元1-2](works/sarashina_embedding/1-2.html)
- [次元1-3](works/sarashina_embedding/1-3.html)
- [次元1-4](works/sarashina_embedding/1-4.html)
- [次元1-5](works/sarashina_embedding/1-5.html)
- [次元2-3](works/sarashina_embedding/2-3.html)

</details>


## 🛠️ 技術的な詳細

### 使用技術

- **言語モデル**: 日本語特化型大規模言語モデル
- **埋め込みベクトル**: 各VTuberの特徴を高次元ベクトルとして表現
- **可視化**: Plotly.jsによるインタラクティブな散布図

### データソース
https://huggingface.co/datasets/Atotti/VTuber-overview-split

### 可視化の見方

1. **点の位置**: 近い位置にある点は、似た特徴を持つVTuberを表します
2. **クラスター**: 密集している領域は、共通の特徴を持つグループを示します
3. **インタラクティブ機能**:
   - マウスホバーで詳細情報を表示
   - ズーム・パンで詳細を探索
   - ダブルクリックでズームをリセット

## 🔗 リンク

- [GitHubリポジトリ](https://github.com/yourusername/vtuber-plot)
- [技術的な実装の詳細（README）](README.md)

