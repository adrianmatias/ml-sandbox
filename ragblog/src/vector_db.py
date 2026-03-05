from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.const import CONST
from src.logger_custom import LOGGER


class VectorDB:
    def __init__(
        self,
    ):
        self.embedding_model = "qwen3-embedding:8b"
        self.collection_name = "collection_ragblog"
        self.persist_directory = CONST.loc.vect_db

        self.store = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=OllamaEmbeddings(model=self.embedding_model),
        )

    def save(self, doc_list: List[Document]) -> None:
        doc_list_count = len(doc_list)
        LOGGER.info(f"{doc_list_count=}")
        Chroma.from_documents(
            documents=doc_list,
            embedding=OllamaEmbeddings(model=self.embedding_model),
            persist_directory=self.persist_directory,
        )

    def load(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=OllamaEmbeddings(model=self.embedding_model),
        )

    def get_vector_db(self, doc_list: List[Document] | None) -> Chroma:
        if doc_list is not None:
            self.save(doc_list=doc_list)
        return self.load()
