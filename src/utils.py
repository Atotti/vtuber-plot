
def save_markdown(content: str, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__=="__main__":
    save_markdown("hoge", "data/DeepResearch/hoge.md")
