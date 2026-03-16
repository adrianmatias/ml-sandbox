from typing import Any, Optional

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.const import CONST, LLM
from src.crawler import Crawler
from src.doc_loader import DocLoader
from src.logger_custom import LOGGER, log_init
from src.vector_db import VectorDB


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
        llm = ChatOpenAI(
            model=self.aug,
            base_url=CONST.api.base_url,
            api_key=CONST.api.api_key,
            max_tokens=4096,
            temperature=0.7,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to "
                    "answer the question. If you don't know the answer, "
                    "just say that you don't know.",
                ),
                ("human", "Question: {question}\n\nContext: {context}\n\nAnswer:"),
            ]
        )

        retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})

        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question: str):
        response = self.chain.invoke(question)
        if response and response.strip():
            return response

        LOGGER.warning(
            "Empty content from ChatOpenAI response; falling back to reasoning_content"
        )
        return self._query_via_openai_fallback(question=question)

    def _query_via_openai_fallback(self, question: str) -> str:
        context = "\n\n".join(self.get_contexts(question=question))
        response = requests.post(
            f"{CONST.api.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {CONST.api.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": str(self.aug),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant for question-answering tasks. "
                            "Use the following pieces of retrieved context to "
                            "answer the question. If you don't know the answer, "
                            "just say that you don't know."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\nContext: {context}\n\nAnswer:"
                        ),
                    },
                ],
                "max_tokens": 1536,
                "temperature": 0.7,
                "no_think": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        completion = response.json()

        message = completion["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if content:
            return content

        reasoning_content = (message.get("reasoning_content") or "").strip()
        return reasoning_content

    def get_contexts(self, question: str):
        """Return retrieved contexts for a question. Public API for evaluators."""
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        return [doc.page_content for doc in docs]

    @staticmethod
    def format_docs(docs):
        doc_intro = "<|retrieved_doc|>"
        return "\n\n".join(doc_intro + doc.page_content for doc in docs)
