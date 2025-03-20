import traceback
import os

from src import vtuber, utils, gpt_search, search_pixiv_dic, plot, plot_plotly

from src.embedding import sarashina_embedding, modernbert_ja_310m, openai_embedding_3_large


def collect_vtuber_info_from_pixiv_dic():
    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Load {len(vtubers)} vtubers!")

    result_list = os.listdir("data/pixiv_dic")

    for v in vtubers:
        name = utils.sanitize_path(v.name)

        if f"{name}.html" in result_list:
            print(f"🦖 {name} is already exist!")
            continue

        print(f"🫠 Start research for {name}")
        try:
            response = search_pixiv_dic.search(name, v.research_prompt)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            print(f"⚠️ {name} research is stop with error.")
            continue

        utils.save_markdown(response, f"data/pixiv_dic/{name}.html")
        print(f"✅ End research for {name} and Saved!")


def collect_vtuber_info_from_gpt_search():
    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Load {len(vtubers)} vtubers!")

    result_list = os.listdir("data/SearchGPT")

    gpt = gpt_search.SearchGPT()

    for v in vtubers:
        name = utils.sanitize_path(v.name)

        if f"{name}.md" in result_list:
            print(f"🦖 {name} is already exist!")
            continue

        print(f"🫠 Start research for {name}")
        try:
            response = gpt.send(v.research_prompt)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            print(f"⚠️ {name} research is stop with error.")
            continue

        utils.save_markdown(response, f"data/SearchGPT/{name}.md")
        print(f"✅ End research for {name} and Saved!")

def vtuber_plot(
        embedding_dir="data/キャラクター性/sarashina_embedding",
        HORIZONTAL_AXIS=1,
        VERTICAL_AXIS=2
    ):
    plot.plot_embeddings_with_pca(
        embedding_dir=embedding_dir,
        HORIZONTAL_AXIS=HORIZONTAL_AXIS,
        VERTICAL_AXIS=VERTICAL_AXIS
        )
    plot_plotly.plot_embeddings_interactive(
        embedding_dir=embedding_dir,
        HORIZONTAL_AXIS=HORIZONTAL_AXIS,
        VERTICAL_AXIS=VERTICAL_AXIS
        )

if __name__ == "__main__":
    # === vtubers.json を生成 ===
    # all_vtubers = vtuber.get_all_vtubers()
    # filtered_vtubers = vtuber.filter_vtubers_by_subscribers(all_vtubers)
    # target_brand_ids = [1, 7, 2, 20, 162, 31, 92, 3, 89, 17, 18, 57, 53, 127, 114]
    # filtered_vtubers = vtuber.filter_vtubers_by_brand_ids(filtered_vtubers, target_brand_ids)
    # vtuber.save_vtubers(filtered_vtubers, "data/filtered_vtubers.json")

    # collect_vtuber_info_from_pixiv_dic()

    # collect_vtuber_info_from_gpt_search()

    sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="キャラクター性")
    # sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="活動内容")
    # sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="コラボ履歴")
    # sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="人間関係")
    # sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="コンテンツのジャンル")
    # sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="活動スタイル")
    # sarashina_embedding.calc_embeddings(dataset_name="Atotti/VTuber-overview-split", split="他のVTuberと比較した時の特徴")


    vtuber_plot(
        embedding_dir="data/キャラクター性/sarashina_embedding",
        HORIZONTAL_AXIS=1,
        VERTICAL_AXIS=2
        )
    # vtuber_plot(
    #     embedding_dir="data/活動内容/sarashina_embedding",
    #     HORIZONTAL_AXIS=1,
    #     VERTICAL_AXIS=2
    #     )
    # vtuber_plot(
    #     embedding_dir="data/コラボ履歴/sarashina_embedding",
    #     HORIZONTAL_AXIS=1,
    #     VERTICAL_AXIS=2
    #     )
    # vtuber_plot(
    #     embedding_dir="data/人間関係/sarashina_embedding",
    #     HORIZONTAL_AXIS=1,
    #     VERTICAL_AXIS=2
    #     )
    # vtuber_plot(
    #     embedding_dir="data/コンテンツのジャンル/sarashina_embedding",
    #     HORIZONTAL_AXIS=1,
    #     VERTICAL_AXIS=2
    #     )
    # vtuber_plot(
    #     embedding_dir="data/活動スタイル/sarashina_embedding",
    #     HORIZONTAL_AXIS=1,
    #     VERTICAL_AXIS=2
    #     )
    # vtuber_plot(
    #     embedding_dir="data/他のVTuberと比較した時の特徴/sarashina_embedding",
    #     HORIZONTAL_AXIS=1,
    #     VERTICAL_AXIS=2
    #     )

    # modernbert_ja_310m.calc_embeddings()

    # openai_embedding_3_large.calc_embeddings()

    # vtuber_plot(embedding_dir="data/modernbert_ja_310m")

    # vtuber_plot(embedding_dir="data/text-embedding-3-large")

    pass
