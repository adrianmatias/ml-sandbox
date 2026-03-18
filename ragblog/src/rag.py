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
        llm = OllamaLLM(model=self.aug)
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
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        return [doc.page_content for doc in docs]

    @staticmethod
    def format_docs(docs):
        doc_intro = "<|retrieved_doc|>"
        return "\n\n".join(doc_intro + doc.page_content for doc in docs)
