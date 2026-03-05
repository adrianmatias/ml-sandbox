from unittest.mock import MagicMock, patch

from src.rag import Rag


@patch("src.rag.VectorDB")
@patch("src.rag.DocLoader")
@patch("src.rag.Crawler")
def test_init_ready_db(mock_crawler, mock_doc_loader, mock_vector_db):
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    rag = Rag(is_ready_vector_db=True)

    mock_crawler.assert_not_called()
    mock_doc_loader.assert_not_called()
    mock_vector_db.assert_called_once()
    mock_vector_db_instance.get_vector_db.assert_called_once_with(doc_list=None)
    assert rag.vector_db == mock_vector_db_instance


@patch("src.rag.VectorDB")
@patch("src.rag.DocLoader")
@patch("src.rag.Crawler")
def test_init_not_ready_db(mock_crawler, mock_doc_loader, mock_vector_db):
    mock_crawler_instance = MagicMock()
    mock_crawler.return_value = mock_crawler_instance
    mock_doc_loader_instance = MagicMock()
    mock_doc_loader.return_value = mock_doc_loader_instance
    mock_doc_loader_instance.load.return_value = "docs"
    mock_vector_db_instance = MagicMock()
    mock_vector_db.return_value = mock_vector_db_instance
    mock_vector_db_instance.get_vector_db.return_value = mock_vector_db_instance

    rag = Rag(is_ready_vector_db=False)

    mock_crawler.assert_called_once_with(post_count_min=2)
    mock_crawler_instance.run.assert_called_once()
    mock_doc_loader.assert_called_once()
    mock_doc_loader_instance.load.assert_called_once()
    mock_vector_db.assert_called_once()
    mock_vector_db_instance.get_vector_db.assert_called_once_with(doc_list="docs")
    assert rag.vector_db == mock_vector_db_instance


def test_format_docs():
    docs = [MagicMock()]
    docs[0].page_content = "content"

    result = Rag.format_docs(docs)

    assert "<|retrieved_doc|>" in result
    assert "content" in result
