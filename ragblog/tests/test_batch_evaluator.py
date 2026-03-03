from unittest.mock import MagicMock

from ragblog.batch_evaluator import evaluate_retrieval, load_test_queries


def test_load_test_queries():
    queries = load_test_queries()
    assert isinstance(queries, list)
    assert len(queries) > 0
    assert all(isinstance(q, str) for q in queries)


def test_evaluate_retrieval():
    mock_evaluator = MagicMock()
    mock_evaluator.llm.invoke.return_value = "Score: 4\nVery relevant."

    mock_docs = [MagicMock(page_content="Test document content")]

    result = evaluate_retrieval(mock_evaluator, "Test question", mock_docs)

    assert "avg_relevance" in result
    assert "precision_at_k" in result
    assert len(result["individual_scores"]) == 1
