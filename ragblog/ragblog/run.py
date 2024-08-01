import os

from ragblog.conf import CONF
from ragblog.crawler import ConfCrawler, Crawler
from ragblog.logger_custom import LoggerCustom
from ragblog.rag_pipeline import (
    JSONLoaderConf,
    RAGChainConf,
    RagPipeline,
    RagPipelineConf,
    TextSplitterConf,
    VectorStoreConf,
)


def main():

    crawler = Crawler(conf_crawler=ConfCrawler(post_count_min=1000))
    crawler.get_url_list()
    crawler.get_post_list()
    crawler.write(path=CONF.path.data)

    rp_conf = RagPipelineConf(
        loader=JSONLoaderConf(
            file_path=os.path.join(CONF.path.data, "blog_all.jsonl"),
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        ),
        splitter=TextSplitterConf(chunk_size=1000, chunk_overlap=10),
        vectorstore=VectorStoreConf(
            embedding_model="nomic-embed-text",
            persist_directory=CONF.path.chroma,
        ),
        ragchain=RAGChainConf(
            prompt_model="rlm/rag-prompt-llama", llm_model="llama3.1"
        ),
        is_db_ready=False,
        is_debug=False,
    )

    pipeline = RagPipeline(conf=rp_conf, logger=LoggerCustom().get_logger())
    response = pipeline.query(
        question="""Describe the relation between Helena and Alejandra. Consider the author's diverse experiences and 
        multifaceted personality, reflecting on traits that are evident across their various blog posts. Provide a 
        detailed and thoughtful response. Ensure your answer 
        is profound and sufficiently long, offering deep insights and personal reflections."""
    )
    print(response)


if __name__ == "__main__":
    main()
