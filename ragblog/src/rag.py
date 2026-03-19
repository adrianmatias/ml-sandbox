import re
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

from src.const import CONST, LLM
from src.crawler import Crawler
from src.doc_loader import DocLoader
from src.logger_custom import log_init
from src.vector_db import VectorDB


class ThinkingOutputParser(StrOutputParser):
    """Strips thinking/reasoning traces from Qwen3.5-27B Unsloth output.

    Handles:
    - <think> blocks
    - "Thinking Process:" sections
    - Unmarked reasoning + double newline + clean answer (newest pattern)
    """

    def parse(self, text: str) -> str:
        cleaned = text

        cleaned = re.sub(
            r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
        )

        cleaned = re.sub(
            r"Thinking Process:.*?(?=\n\n[A-Z]|\Z)",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        cleaned = re.sub(r"^.*?\n\n(?=[A-Z])", "", cleaned, flags=re.DOTALL)

        cleaned = re.sub(r"\u16ee.*?\u16ed", "", cleaned, flags=re.DOTALL)

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()


@log_init
class Rag:
    def __init__(self, is_overwrite_index: bool = False, aug: Optional[LLM] = None):
        self.aug = aug or CONST.model.aug

        vdb = VectorDB()
        if is_overwrite_index or not vdb.persist_directory.exists():
            crawler = Crawler(post_count_min=100)
            crawler.run()
            doc_list = DocLoader().load()
        else:
            doc_list = None

        self.vector_db = vdb.get_vector_db(doc_list=doc_list)
        self.chain = self.create_chain()

    def create_chain(self) -> Any:
        # num_ctx = 1024
        num_ctx = None

        llm = OllamaLLM(model=self.aug, num_ctx=num_ctx)
        prompt = PromptTemplate.from_template(
            """human

[INST]<<SYS>> You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.<</SYS>> 

Question: {question} 

Context: {context} 

Answer: [/INST]""",
        )
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})

        is_think_on = self.aug == CONST.model.aug.QWEN_3_5_27B_Q2
        output_parser = ThinkingOutputParser() if is_think_on else StrOutputParser()

        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )

    def query(self, question: str):
        return self.chain.invoke(question)

    def get_contexts(self, question: str):
        """Return retrieved contexts for a question. Public API for evaluators."""
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        return [doc.page_content for doc in docs]

    @staticmethod
    def format_docs(docs):
        doc_intro = "<|retrieved_doc|>"
        return "\n\n".join(doc_intro + doc.page_content for doc in docs)
