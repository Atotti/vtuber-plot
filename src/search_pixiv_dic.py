import requests
import time
from bs4 import BeautifulSoup


def search(name: str, prompt: str) -> str:
    BASE_URL = "https://dic.pixiv.net/"
    search_url = BASE_URL + "search?query=" + name

    max_retries = 3
    delay_seconds = 5

    for i in range(max_retries):
        search_response = requests.get(search_url)
        if search_response.status_code == 200:
            print(f"    ✅ Get search results {name}")
            break
        else:
            if i < max_retries - 1:
                time.sleep(delay_seconds)
    else:
        raise Exception(
            f"Failed to retrieve data from {search_url} after {max_retries} attempts."
        )

    soup = BeautifulSoup(search_response.text, "html.parser")
    search_results = soup.find_all("article")[0]
    target_article = search_results.find_all("a")[1]

    target_article_url = target_article.get("href")

    article_url = BASE_URL + target_article_url

    for i in range(max_retries):
        article_response = requests.get(article_url)
        if article_response.status_code == 200:
            print(f"    ✅ Get {name} page")
            break
        else:
            if i < max_retries - 1:
                time.sleep(delay_seconds)
    else:
        raise Exception(
            f"Failed to retrieve data from {article_url} after {max_retries} attempts."
        )

    soup = BeautifulSoup(article_response.text, "html.parser")
    article = soup.find("article")

    return article.decode_contents()
