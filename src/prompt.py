
def gen_prompt(name: str, template_path: str = "prompts/research.txt") -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(name=name)

if __name__=="__main__":
    print(gen_prompt("hoge"))
