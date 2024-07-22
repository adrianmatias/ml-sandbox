from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragblog.conf import CONF
from ragblog.logger_custom import LoggerCustom
from ragblog.rag_pipeline import (
    JSONLoaderConf,
    RAGChainConf,
    RagPipeline,
    RagPipelineConf,
    TextSplitterConf,
    VectorStoreConf,
)


# Mock the dependencies
@pytest.fixture
def mock_loader():
    loader = MagicMock(JSONLoader)
    loader.load.return_value = [{"text": "document1"}, {"text": "document2"}]
    return loader


@pytest.fixture
def mock_splitter():
    splitter = MagicMock(RecursiveCharacterTextSplitter)
    splitter.split_documents.return_value = ["split1", "split2", "split3"]
    return splitter


@pytest.fixture
def mock_vectorstore():
    vectorstore = MagicMock(Chroma)
    vectorstore.as_retriever.return_value = MagicMock()
    return vectorstore


@pytest.fixture
def rag_pipeline_conf():
    return RagPipelineConf(
        loader=JSONLoaderConf(
            file_path="data/2024-07-22_17-12-41.jsonl",
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        ),
        splitter=TextSplitterConf(chunk_size=1000, chunk_overlap=200),
        vectorstore=VectorStoreConf(
            embedding_model="shaw/dmeta-embedding-zh",
            persist_directory=CONF.path.chroma,
        ),
        ragchain=RAGChainConf(prompt_model="rlm/rag-prompt", llm_model="llama3"),
        is_db_ready=False,
    )


@pytest.fixture
def logger():
    return LoggerCustom().get_logger()


@pytest.fixture
def rag_pipeline():
    conf = RagPipelineConf(
        loader=JSONLoaderConf(
            file_path="data/2024-07-22_17-12-41.jsonl",
            jq_schema=".text",
            text_content=False,
            json_lines=True,
        ),
        splitter=TextSplitterConf(chunk_size=1000, chunk_overlap=200),
        vectorstore=VectorStoreConf(
            embedding_model="shaw/dmeta-embedding-zh",
            persist_directory=CONF.path.chroma,
        ),
        ragchain=RAGChainConf(prompt_model="rlm/rag-prompt", llm_model="llama3"),
        is_db_ready=False,
    )

    return RagPipeline(conf=conf, logger=LoggerCustom().get_logger())


def test_load_documents(mock_loader, rag_pipeline):
    rag_pipeline.load_documents = mock_loader.load
    documents = rag_pipeline.load_documents()

    assert len(documents) == 2
    assert documents[0]["text"] == "document1"


def test_split_documents(mock_splitter, rag_pipeline):
    rag_pipeline.split_documents = mock_splitter.split_documents
    splits = rag_pipeline.split_documents([{"text": "document1"}])

    assert len(splits) == 3
    assert splits[0] == "split1"


def test_create_vectorstore(mock_vectorstore, rag_pipeline):
    rag_pipeline.create_vectorstore = mock_vectorstore.from_documents
    splits = ["split1", "split2", "split3"]
    vectorstore = rag_pipeline.create_vectorstore(splits)

    assert vectorstore is not None


def test_create_rag_chain(mock_vectorstore, rag_pipeline):
    retriever = MagicMock()
    prompt = MagicMock()
    rag_pipeline.create_rag_chain = MagicMock(return_value="rag_chain")
    rag_chain = rag_pipeline.create_rag_chain(retriever, prompt)

    assert rag_chain == "rag_chain"
