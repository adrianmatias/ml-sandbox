import os
from typing import List

import requests
from bs4 import BeautifulSoup

from ragblog.const import CONST
from ragblog.logger_custom import LoggerCustom
from ragblog.post import Post, PostEmpty

LOGGER = LoggerCustom().get_logger()


class Crawler:
    def __init__(self, post_count_min: int):
        self.url_list: List[str] = []
        self.post_list: List[Post] = []

        self.url = "https://delightfulobservaciones.blogspot.com/"
        self.post_count_min = post_count_min

        LOGGER.info(f"self.__dict__: {self.__dict__}")

    def run(self):
        self.get_url_list()
        self.get_post_list()
        self.write(path=CONST.loc.data)

    def get_url_list(self) -> None:
        current_url = self.url
        url_list = []

        while current_url:
            if len(url_list) >= self.post_count_min:
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
                LOGGER.info(f"url: {url}")
                url_list.append(url)

            next_button = soup.find(
                "a", class_="blog-pager-older-link flat-button ripple"
            )
            if next_button:
                current_url = next_button["href"]
                LOGGER.info(f"Next page URL: {current_url}")
            else:
                LOGGER.info("No more pages found.")
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
        path_file = os.path.join(path, "blog.jsonl")
        LOGGER.info(f"path_file: {path_file}")
        with open(file=path_file, mode="w") as file_write:
            content = "\n".join(map(str, self.post_list))
            file_write.write(content)
