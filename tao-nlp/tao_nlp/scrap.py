from dataclasses import dataclass

import pandas as pd
import requests
from bs4 import BeautifulSoup


@dataclass
class Article:
    title: str
    url: str
    text: str


def fetch_article(url: str) -> Article:
    """
    Fetches an article from a given URL and returns an Article object.

    :param url: The URL of the article to fetch.
    :return: An Article object containing the title, URL, and text of the fetched article.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    title = soup.find(id="firstHeading").text
    text = "".join(paragraph.text for paragraph in soup.find_all("p"))

    return Article(title=title, url=url, text=text)


def main():
    """
    The main function that initializes the URL list and fetches articles.
    """
    url_list = [
        "https://en.wikipedia.org/wiki/Philosophy",
        "https://en.wikipedia.org/wiki/Definitions_of_philosophy",
    ]

    articles = [fetch_article(url) for url in url_list]
    df = pd.DataFrame(articles)

    print(df.info())
    print(df)


if __name__ == "__main__":
    main()
