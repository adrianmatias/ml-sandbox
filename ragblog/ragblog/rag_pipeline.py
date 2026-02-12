from dataclasses import dataclass
from logging import Logger
from typing import Any, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_core.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class JSONLoaderConf:
    file_path: str
    jq_schema: str
    text_content: bool
    json_lines: bool


@dataclass
class TextSplitterConf:
    chunk_size: int
    chunk_overlap: int


@dataclass
class VectorStoreConf:
    embedding_model: str
    persist_directory: str


@dataclass
class RAGChainConf:
    prompt_model: str
    llm_model: str


@dataclass
class RagPipelineConf:
    loader: JSONLoaderConf
    splitter: TextSplitterConf
    vectorstore: VectorStoreConf
    ragchain: RAGChainConf
    is_db_ready: bool
    is_debug: bool


class RagPipeline:
    def __init__(self, conf: RagPipelineConf, logger: Logger):
        self.conf = conf
        self.logger = logger

        self.logger.info(f"{self.conf}")
        set_debug(self.conf.is_debug)

    def load_documents(self) -> List[Document]:

        loader = JSONLoader(
            file_path=self.conf.loader.file_path,
            jq_schema=self.conf.loader.jq_schema,
            text_content=self.conf.loader.text_content,
            json_lines=self.conf.loader.json_lines,
        )
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.conf.splitter.chunk_size,
            chunk_overlap=self.conf.splitter.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, splits: List[Document]) -> Any:
        split_count = len(splits)
        self.logger.info(f"split_count: {split_count}")
        return Chroma.from_documents(
            documents=splits,
            embedding=OllamaEmbeddings(model=self.conf.vectorstore.embedding_model),
            persist_directory=self.conf.vectorstore.persist_directory,
        )

    def load_vectorstore(self):
        return Chroma(
            persist_directory=self.conf.vectorstore.persist_directory,
            embedding_function=OllamaEmbeddings(
                model=self.conf.vectorstore.embedding_model
            ),
        )

    def create_rag_chain(self, retriever, prompt) -> Any:
        llm = OllamaLLM(model=self.conf.ragchain.llm_model)
        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def get_vectorstore(self):
        if self.conf.is_db_ready:
            return self.load_vectorstore()

        else:
            doc_list = self.load_documents()
            splits = self.split_documents(doc_list)
            return self.create_vectorstore(splits)

    def query(self, question: str):

        return self.create_rag_chain(
            retriever=self.get_vectorstore().as_retriever(k=10),
            prompt=PromptTemplate.from_template(
                """human

[INST]<<SYS>> You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.<</SYS>> 

Question: {question} 

Context: {context} 

Answer: [/INST]""",
            ),
        ).invoke(question)

    @staticmethod
    def format_docs(docs):
        for doc in docs:
            print(doc)
        doc_intro = "<|retrieved_doc|>"
        return "\n\n".join(doc_intro + doc.page_content for doc in docs)
