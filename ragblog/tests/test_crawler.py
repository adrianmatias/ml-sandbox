from ragblog.conf import ConfCrawler
from ragblog.crawler import Crawler

CRAWLER = Crawler(conf_crawler=ConfCrawler())


def test_get_url_list():

    CRAWLER.get_url_list()

    assert len(CRAWLER.url_list) >= CRAWLER.conf.post_count_min
    assert all(map(lambda url: url.startswith(CRAWLER.conf.url), CRAWLER.url_list))


def test_get_post_list():
    CRAWLER.url_list = [
        "https://delightfulobservaciones.blogspot.com/2019/06/poco-antes-de-helena.html",
        "https://delightfulobservaciones.blogspot.com/2023/10/poco-antes-de-alejandra.html",
        "https://github.com/",
    ]
    CRAWLER.get_post_list()
    assert len(CRAWLER.post_list) == 2
    assert CRAWLER.post_list[0].title == "Poco antes de Helena"
