from unittest.mock import MagicMock

import pytest

from ragblog.logger_custom import LoggerCustom
from ragblog.rag_evaluator import RagEvaluator, RagEvaluatorConf


@pytest.fixture
def evaluator_conf():
    return RagEvaluatorConf(
        llm_model="llama3",
        relevance_weight=1.0,
        faithfulness_weight=1.0,
        coherence_weight=0.5,
    )


@pytest.fixture
def logger():
    return LoggerCustom().get_logger()


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.return_value = "Score: 4\nThis answer is highly relevant."
    return llm


@pytest.fixture
def evaluator(evaluator_conf, logger, mock_llm):
    eval_obj = RagEvaluator(evaluator_conf, logger)
    eval_obj.llm = mock_llm
    return eval_obj


def test_evaluate(evaluator):
    question = "What is the capital of France?"
    answer = "Paris"
    context = "France is a country in Europe. Its capital is Paris."

    result = evaluator.evaluate(question, answer, context)

    assert "relevance" in result
    assert "faithfulness" in result
    assert "coherence" in result
    assert "overall_score" in result
    assert isinstance(result["overall_score"], float)


def test_evaluate_metric_relevance(evaluator, mock_llm):
    mock_llm.return_value = "Score: 5\nVery relevant."
    result = evaluator._evaluate_metric("relevance", "Question", "Answer")
    assert result["score"] == 5.0
    assert "Very relevant" in result["explanation"]


def test_evaluate_metric_faithfulness(evaluator, mock_llm):
    mock_llm.return_value = "Score: 3\nSomewhat faithful."
    result = evaluator._evaluate_metric("faithfulness", "Question", "Answer", "Context")
    assert result["score"] == 3.0
    assert "Somewhat faithful" in result["explanation"]


def test_evaluate_metric_coherence(evaluator, mock_llm):
    mock_llm.return_value = "Score: 4\nCoherent answer."
    result = evaluator._evaluate_metric("coherence", "Question", "Answer")
    assert result["score"] == 4.0
    assert "Coherent answer" in result["explanation"]


def test_extract_score(evaluator):
    assert evaluator._extract_score("Score: 4") == 4.0
    assert evaluator._extract_score("5 stars") == 5.0
    assert evaluator._extract_score("No score here") == 3.0  # default
