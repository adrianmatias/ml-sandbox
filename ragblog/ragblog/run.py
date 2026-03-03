import json
import os

from ragblog.conf import CONF
from ragblog.crawler import ConfCrawler, Crawler
from ragblog.logger_custom import LoggerCustom
from ragblog.rag_evaluator import RagEvaluatorConf
from ragblog.rag_pipeline import (
    JSONLoaderConf,
    RAGChainConf,
    RagPipeline,
    RagPipelineConf,
    TextSplitterConf,
    VectorStoreConf,
)


def main():
    logger = LoggerCustom().get_logger()

    crawler = Crawler(conf_crawler=ConfCrawler(post_count_min=1000))
    crawler.get_url_list()
    crawler.get_post_list()
    crawler.write(path=CONF.path.data)

    evaluator_conf = RagEvaluatorConf(
        llm_model="gpt-oss:20b",
        relevance_weight=1.0,
        faithfulness_weight=1.0,
        coherence_weight=0.5,
    )

    rp_conf = RagPipelineConf(
        loader=JSONLoaderConf(
            file_path=os.path.join(CONF.path.data, "blog.jsonl"),
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        ),
        splitter=TextSplitterConf(chunk_size=1000, chunk_overlap=10),
        vectorstore=VectorStoreConf(
            embedding_model="qwen3-embedding:8b",
            persist_directory=CONF.path.chroma,
        ),
        ragchain=RAGChainConf(
            prompt_model="rlm/rag-prompt-llama", llm_model="gpt-oss:20b"
        ),
        is_db_ready=False,
        is_debug=False,
        evaluator=evaluator_conf,
    )

    pipeline = RagPipeline(conf=rp_conf, logger=logger)

    question = """Describe the relation between Helena and Alejandra.
        Consider the author's diverse experiences and multifaceted personality,
        reflecting on traits that are evident across their various blog posts.
        Provide a detailed and thoughtful response.
        Ensure your answer is profound and sufficiently long, 
        offering deep insights and personal reflections."""
    answer = pipeline.query(question=question)
    print(answer)

    evaluation = pipeline.evaluate_answer(question=question, answer=answer, context="")
    logger.info(f"{evaluation=}")

    output_path = os.path.join(CONF.path.data, "output.md")
    logger.info(f"{output_path}")
    os.makedirs(CONF.path.data, exist_ok=True)

    code_bound = "```"
    with open(output_path, "w") as f:
        content = "\n\n".join(
            [
                "## question",
                question,
                "## answer",
                answer,
                "## evaluation",
                code_bound,
                json.dumps(evaluation, indent=4),
                code_bound,
            ]
        )
        f.write(content)


if __name__ == "__main__":
    main()
