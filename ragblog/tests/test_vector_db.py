from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vector_db import VectorDB


@pytest.fixture
def vector_db():
    return VectorDB()


def test_init(vector_db):
    assert vector_db.model == "qwen3-embedding:8b"
    assert vector_db.collection_name == "collection_ragblog"


@patch("src.vector_db.Chroma")
@patch("src.vector_db.OllamaEmbeddings")
def test_save(mock_ollama, mock_chroma, vector_db):
    mock_embedding_instance = MagicMock()
    mock_ollama.return_value = mock_embedding_instance
    mock_chroma.from_documents = MagicMock()

    docs = [Document(page_content="test")]
    vector_db.save(docs)

    mock_chroma.from_documents.assert_called_once_with(
        documents=docs,
        embedding=mock_embedding_instance,
        persist_directory=vector_db.persist_directory,
        collection_name=vector_db.collection_name,
    )


@patch("src.vector_db.Chroma")
@patch("src.vector_db.OllamaEmbeddings")
def test_load(mock_ollama, mock_chroma, vector_db):
    mock_embedding_instance = MagicMock()
    mock_ollama.return_value = mock_embedding_instance
    mock_chroma_instance = MagicMock()
    mock_chroma.return_value = mock_chroma_instance

    result = vector_db.load()

    mock_chroma.assert_called_with(
        persist_directory=vector_db.persist_directory,
        collection_name=vector_db.collection_name,
        embedding_function=mock_embedding_instance,
    )
    assert result == mock_chroma_instance


@patch.object(VectorDB, "save")
@patch.object(VectorDB, "load")
def test_get_vector_db_with_docs(mock_load, mock_save, vector_db):
    mock_load.return_value = "loaded_db"
    docs = [Document(page_content="test")]

    result = vector_db.get_vector_db(docs)

    mock_save.assert_called_once_with(doc_list=docs)
    mock_load.assert_called_once()
    assert result == "loaded_db"


@patch.object(VectorDB, "save")
@patch.object(VectorDB, "load")
def test_get_vector_db_without_docs(mock_load, mock_save, vector_db):
    mock_load.return_value = "loaded_db"

    result = vector_db.get_vector_db(None)

    mock_save.assert_not_called()
    mock_load.assert_called_once()
    assert result == "loaded_db"
