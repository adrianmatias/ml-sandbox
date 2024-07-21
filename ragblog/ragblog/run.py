from ragblog import conf
from ragblog.conf import ConfCrawler
from ragblog.crawler import Crawler


def main():

    crawler = Crawler(conf_crawler=ConfCrawler())
    crawler.get_url_list()
    crawler.get_post_list()
    crawler.write(path=conf.Path().data)


if __name__ == "__main__":
    main()
