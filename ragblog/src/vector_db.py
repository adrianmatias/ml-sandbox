import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.const import CONST
from src.logger_custom import LOGGER, log_init
from src.ollama_embeddings import OllamaEmbeddings


@log_init
class VectorDB:
    def __init__(
        self,
    ):
        self.model = CONST.model.emb
        self.persist_directory = CONST.loc.vect_db
        self.collection_name = "collection_ragblog"

    def _make_embeddings(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=self.model,
            base_url=CONST.api.emb_url,
        )

    def save(self, doc_list: List[Document]) -> None:
        doc_list_count = len(doc_list)
        LOGGER.info(f"{doc_list_count=}")
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        Chroma.from_documents(
            documents=doc_list,
            embedding=self._make_embeddings(),
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

    def load(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            embedding_function=self._make_embeddings(),
        )

    def get_vector_db(self, doc_list: List[Document] | None) -> Chroma:
        if doc_list is not None:
            self.save(doc_list=doc_list)
        return self.load()
