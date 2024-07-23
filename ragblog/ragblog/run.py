import os

from ragblog.conf import CONF_BLOG_FULL
from ragblog.crawler import Crawler, ConfCrawler
from ragblog.logger_custom import LoggerCustom
from ragblog.rag_pipeline import RagPipelineConf, JSONLoaderConf, TextSplitterConf, VectorStoreConf, RAGChainConf, \
    RagPipeline


def main():

    crawler = Crawler(conf_crawler=ConfCrawler(post_count_min=1000))
    crawler.get_url_list()
    crawler.get_post_list()
    crawler.write(path=CONF_BLOG_FULL.path.data)

    conf = CONF_BLOG_FULL
    rp_conf = RagPipelineConf(
        loader=JSONLoaderConf(
            file_path=os.path.join(conf.path.data, "blog_all.jsonl"),
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        ),
        splitter=TextSplitterConf(chunk_size=1000, chunk_overlap=200),
        vectorstore=VectorStoreConf(
            embedding_model="shaw/dmeta-embedding-zh",
            persist_directory=conf.path.chroma,
        ),
        ragchain=RAGChainConf(prompt_model="rlm/rag-prompt", llm_model="llama3"),
        is_db_ready=False,
    )
    pipeline = RagPipeline(conf=rp_conf, logger=LoggerCustom().get_logger())
    response = pipeline.query(
        question="""Describe the journey of fatherhood as an adventure. Consider the author's diverse experiences and 
        multifaceted personality, reflecting on traits that are evident across their various blog posts. Provide a 
        detailed and thoughtful response, capturing the complexities and joys of being a father. Ensure your answer 
        is profound and sufficiently long, offering deep insights and personal reflections."""
    )
    print(response)


if __name__ == "__main__":
    main()
