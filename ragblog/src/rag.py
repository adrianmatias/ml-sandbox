from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

from src.const import CONST
from src.crawler import Crawler
from src.doc_loader import DocLoader
from src.vector_db import VectorDB


class Rag:
    def __init__(self, is_ready_vector_db: bool):
        if is_ready_vector_db:
            doc_list = None
        else:
            crawler = Crawler(post_count_min=2)
            crawler.run()
            doc_list = DocLoader().load()

        self.vector_db = VectorDB().get_vector_db(doc_list=doc_list)
        self.chain = self.create_chain()

    def create_chain(self) -> Any:
        llm = OllamaLLM(model=CONST.model.rag)
        prompt = PromptTemplate.from_template(
            """human

[INST]<<SYS>> You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.<</SYS>> 

Question: {question} 

Context: {context} 

Answer: [/INST]""",
        )
        retriever = self.vector_db.as_retriever(k=10)

        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query(self, question: str):
        return self.chain.invoke(question)

    def get_contexts(self, question: str):
        """Return retrieved contexts for a question. Public API for evaluators."""
        retriever = self.vector_db.as_retriever(k=10)
        docs = retriever.invoke(question)
        return [doc.page_content for doc in docs]

    @staticmethod
    def format_docs(docs):
        doc_intro = "<|retrieved_doc|>"
        return "\n\n".join(doc_intro + doc.page_content for doc in docs)
