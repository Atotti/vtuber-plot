import re


def save_markdown(content: str, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)


def sanitize_path(text: str) -> str:
    sanitized = re.sub(r"にじさんじ", " ", text)
    sanitized = re.sub(r'[\\/:\*\?"<>|\-【】]', " ", sanitized)
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    return sanitized.strip()


if __name__ == "__main__":
    save_markdown("hoge", "data/DeepResearch/hoge.md")
