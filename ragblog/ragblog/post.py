from __future__ import annotations

import json
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass
class Post:
    title: str
    text: str

    def __str__(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_url(cls, url: str) -> Post:
        post_response = requests.get(url)
        if post_response.status_code != 200:
            print(f"Failed to load post {url}")
            return PostEmpty()

        post_soup = BeautifulSoup(post_response.content, "html.parser")

        content_div = post_soup.find("div", class_="post-body")
        text = content_div.get_text(separator="\n").strip() if content_div else ""
        if text == "":
            return PostEmpty()

        return cls(
            title=post_soup.find("h3", class_="entry-title").text.strip(), text=text
        )


class PostEmpty(Post):
    def __init__(self):
        self.title = ""
        self.text = ""
