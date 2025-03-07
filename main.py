import vtuber
import deep_research
import utils

def main():
    vtubers_json_path = "data/filterd_vtubers.json"
    vtubers = vtuber.load_vtubers(vtubers_json_path)

    print(f"🎁 Load {len(vtubers)} vtubers!")

    for v in vtubers:
        print(f"🫠 Start research for {v.name}")
        prompt = v.research_prompt
        try:
            deep_research.deep_research(prompt)
        except Exception:
            print(f"⚠️ {v.name} research is stop with error.")
            continue

        utils.save_markdown(prompt, f"data/DeepResearch/{v.name}.html")
        print(f"✅ End research for {v.name} and Saved!")
