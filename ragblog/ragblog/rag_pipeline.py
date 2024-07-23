import os
from dataclasses import dataclass
from logging import Logger
from typing import Any, List

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragblog.conf import CONF_BLOG_FULL
from ragblog.logger_custom import LoggerCustom


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


class RagPipeline:
    def __init__(self, conf: RagPipelineConf, logger: Logger):
        self.Conf = conf
        self.logger = logger

        self.logger.info(f"{self.Conf}")

    def load_documents(self) -> List[Document]:

        loader = JSONLoader(
            file_path=self.Conf.loader.file_path,
            jq_schema=self.Conf.loader.jq_schema,
            text_content=self.Conf.loader.text_content,
            json_lines=self.Conf.loader.json_lines,
        )
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.Conf.splitter.chunk_size,
            chunk_overlap=self.Conf.splitter.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, splits: List[Document]) -> Any:
        return Chroma.from_documents(
            documents=splits,
            embedding=OllamaEmbeddings(model=self.Conf.vectorstore.embedding_model),
            persist_directory=self.Conf.vectorstore.persist_directory,
        )

    def load_vectorstore(self):
        return Chroma(
            persist_directory=self.Conf.vectorstore.persist_directory,
            embedding_function=OllamaEmbeddings(
                model=self.Conf.vectorstore.embedding_model
            ),
        )

    def create_rag_chain(self, retriever, prompt) -> Any:
        llm = Ollama(model=self.Conf.ragchain.llm_model)
        return (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def get_vectorstore(self):
        if self.Conf.is_db_ready:
            return self.load_vectorstore()

        else:
            doc_list = self.load_documents()
            splits = self.split_documents(doc_list)
            return self.create_vectorstore(splits)

    def query(self, question: str):

        return self.create_rag_chain(
            retriever=self.get_vectorstore().as_retriever(),
            prompt=hub.pull(self.Conf.ragchain.prompt_model),
        ).invoke(question)

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


def main():

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
