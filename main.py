import traceback
import os

from src import vtuber, deep_research, utils


def collect_vtuber_info():
    vtubers_json_path = "data/filtered_vtubers.json"
    vtubers = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Load {len(vtubers)} vtubers!")

    result_list = os.listdir("data/DeepResearch")

    for v in vtubers:
        name = utils.sanitize_path(v.name)

        if f"{name}.html" in result_list:
            print(f"🦖 {name} is already exist!")
            continue

        print(f"🫠 Start research for {name}")
        try:
            response = deep_research.deep_research(name, v.research_prompt)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            print(f"⚠️ {name} research is stop with error.")
            continue

        utils.save_markdown(response, f"data/DeepResearch/{name}.html")
        print(f"✅ End research for {name} and Saved!")


