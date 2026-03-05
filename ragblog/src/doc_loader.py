from typing import List

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.const import CONST


class DocLoader:
    def __init__(
        self,
    ):
        self.a = "a"

    def load(self) -> List[Document]:
        loader = JSONLoader(
            file_path=CONST.loc.data / "blog.jsonl",
            jq_schema=".text",
            json_lines=True,
        )
        doc_list = loader.load()

        return self.split(doc_list=doc_list)

    @staticmethod
    def split(doc_list: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=10,
        )
        return splitter.split_documents(documents=doc_list)
