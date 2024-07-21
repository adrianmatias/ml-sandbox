import os
from datetime import datetime
from typing import List

import requests
from bs4 import BeautifulSoup

from ragblog import conf
from ragblog.conf import ConfCrawler
from ragblog.logger_custom import LoggerCustom
from ragblog.post import Post, PostEmpty

LOGGER = LoggerCustom().get_logger()


class Crawler:
    def __init__(self, conf_crawler: ConfCrawler):
        self.conf = conf_crawler
        self.url_list: List[str] = []
        self.post_list: List[Post] = []
        LOGGER.info(f"self.__dict__: {self.__dict__}")

    def get_url_list(self) -> None:
        current_url = self.conf.url
        url_list = []

        while current_url:
            if len(url_list) >= self.conf.post_count_min:
                break

            LOGGER.info(f"Fetching {current_url}")
            response = requests.get(current_url)
            if response.status_code != 200:
                LOGGER.info(f"Failed to load page {current_url}")
                break

            soup = BeautifulSoup(response.content, "html.parser")

            # Find all post urls on the current page
            posts = soup.find_all("h3", class_="post-title entry-title")
            for post in posts:
                url = post.find("a")["href"]
                url_list.append(url)

            # Find the URL for the next page
            next_button = soup.find("a", class_="blog-pager-older-url")
            if next_button:
                current_url = next_button["href"]
            else:
                current_url = None

        self.url_list = sorted(list(set(url_list)))

    def get_post_list(self) -> None:
        self.post_list = list(
            filter(
                lambda post: not isinstance(post, PostEmpty),
                map(Post.from_url, self.url_list),
            )
        )

    def write(self, path: str):
        path_file = os.path.join(
            path, f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt"
        )
        LOGGER.info(f"path_file: {path_file}")
        with open(file=path_file, mode="w") as file_write:
            content = "\n".join(map(str, self.post_list))
            file_write.write(content)


def main():

    crawler = Crawler(conf_crawler=ConfCrawler())
    crawler.get_url_list()
    crawler.get_post_list()

    crawler.write(path=conf.Path().data)


if __name__ == "__main__":
    main()
