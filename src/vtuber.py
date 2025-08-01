import requests
from bs4 import BeautifulSoup
from typing import List
import dataclasses
import time
import json

from src.prompt import gen_prompt

BASE_URL = "https://www.vstats.jp/brands/"


@dataclasses.dataclass
class VTuber:
    name: str
    subscribers: int
    brand_id: int
    research_prompt: str


def get_brand_vtuber_list(brand_id: int) -> List[VTuber]:
    brand_url = f"{BASE_URL}{brand_id}"

    max_retries = 5
    delay_seconds = 0.1

    for i in range(max_retries):
        response = requests.get(brand_url)
        if response.status_code == 200:
            print(f"✅ Get brand/{brand_id} page")
            break
        else:
            if i < max_retries - 1:
                time.sleep(delay_seconds)
    else:
        raise Exception(
            f"Failed to retrieve data from {brand_url} after {max_retries} attempts."
        )

    soup = BeautifulSoup(response.text, "html.parser")
    vtubers: List[VTuber] = []
    vtuber_elements = soup.find_all("div", class_="col")

    for vtuber_element in vtuber_elements:
        # リンク要素を取得
        a_tag = vtuber_element.find("a", class_="stretched-link link-success")
        if not a_tag:
            continue

        # 名前を取得
        name = a_tag.get_text(strip=True)

        # 登録者数取得
        subscribers_div = vtuber_element.find_all("div", class_="fs-5")[-1]
        subscribers_str = subscribers_div.get_text(strip=True)
        subscribers = int(subscribers_str.replace(",", ""))

        # データクラスにまとめてリストに追加
        vtubers.append(
            VTuber(
                name=name,
                subscribers=subscribers,
                brand_id=brand_id,
                research_prompt=gen_prompt(name),
            )
        )

    if len(vtubers) == 0:
        raise Exception("⚠️ No data!")

    return vtubers


def get_all_vtubers() -> List[VTuber]:
    vtubers: List[VTuber] = []

    for brand_id in range(1, 200):
        try:
            brand_vtubers = get_brand_vtuber_list(brand_id)
        except Exception:
            continue
        vtubers += brand_vtubers

        print(f"🦖 brand {brand_id} in {len(brand_vtubers)} livers!")

    return vtubers


def save_vtubers(vtubers: List[VTuber], save_path: str):
    data = [dataclasses.asdict(v) for v in vtubers]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"✅ Saved VTuber list {save_path}!")


def filter_vtubers_by_subscribers(
    vtubers: List[VTuber], subscribers: int = 30_000
) -> List[VTuber]:
    filtered_vtubers: List[VTuber] = []
    for v in vtubers:
        if v.subscribers >= subscribers:
            filtered_vtubers.append(v)
    print(f"🦖 Filterd to {len(filtered_vtubers)} by subscribers!")
    return filtered_vtubers


def filter_vtubers_by_brand_ids(
    vtubers: List[VTuber], brand_ids: List[int]
) -> List[VTuber]:
    filtered_vtubers: List[VTuber] = []
    for v in vtubers:
        if v.brand_id in brand_ids:
            filtered_vtubers.append(v)
    print(f"🦖 Filterd to {len(filtered_vtubers)} by brand_ids!")
    return filtered_vtubers


def load_vtubers(load_path: str) -> List[VTuber]:
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [VTuber(**item) for item in data]


if __name__ == "__main__":
    all_vtubers = get_all_vtubers()
    filtered_vtubers = filter_vtubers_by_subscribers(all_vtubers)
    target_brand_ids = [1, 7, 2, 20, 162, 31, 92, 3, 89, 17, 18, 57]
    filtered_vtubers = filter_vtubers_by_brand_ids(filtered_vtubers, target_brand_ids)
    save_vtubers(filtered_vtubers, "data/filtered_vtubers.json")
