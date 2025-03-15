import traceback
import os

from src import vtuber, utils, gpt_search, search_pixiv_dic, plot, plot_plotly

from src.embedding import sarashina_embedding, modernbert_ja_310m


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


if __name__ == "__main__":
    # === vtubers.json を生成 ===
    # all_vtubers = vtuber.get_all_vtubers()
    # filtered_vtubers = vtuber.filter_vtubers_by_subscribers(all_vtubers)
    # target_brand_ids = [1, 7, 2, 20, 162, 31, 92, 3, 89, 17, 18, 57, 53, 127, 114]
    # filtered_vtubers = vtuber.filter_vtubers_by_brand_ids(filtered_vtubers, target_brand_ids)
    # vtuber.save_vtubers(filtered_vtubers, "data/filtered_vtubers.json")

    # collect_vtuber_info_from_pixiv_dic

    # collect_vtuber_info_from_gpt_search()

    # sarashina_embedding.calc_embeddings()

    # modernbert_ja_310m.calc_embeddings()

    # plot.plot_embeddings_with_pca(embedding_dir="data/sarashina_embedding")
    # plot_plotly.plot_embeddings_interactive(embedding_dir="data/sarashina_embedding")

    # plot.plot_embeddings_with_pca(embedding_dir="data/modernbert_ja_310m")
    # plot_plotly.plot_embeddings_interactive(embedding_dir="data/modernbert_ja_310m")
    pass
