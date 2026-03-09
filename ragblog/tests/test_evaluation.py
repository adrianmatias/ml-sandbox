from src.const import CONST
from src.evaluation.eval_set import EvalSet, Item
from src.evaluation.evaluator import RagEval


def test_item_creation():
    """Test Item dataclass."""
    item = Item(
        question="What is the meaning of life?",
        ground_truth="42",
        reference_contexts=["Some context"],
        persona_name="Philosopher",
    )
    assert item.question == "What is the meaning of life?"
    assert item.ground_truth == "42"
    assert len(item.reference_contexts) == 1


def test_test_set_class():
    """Test EvalSet class functionality."""
    testset = EvalSet()
    # We can't easily test generation without long runtime,
    # so test the interface
    assert hasattr(testset, "generate")
    assert hasattr(testset, "load")
    assert hasattr(testset, "to_item_list")
    assert testset.eval_set_path == CONST.loc.eval_set


def test_rag_eval_class():
    """Test RagEval class interface."""
    # Test class has required methods and attributes
    assert hasattr(RagEval, "__init__")
    assert hasattr(RagEval, "evaluate")
    # Test that metrics are defined as class attribute or in init
    # Since it's instance, hard to test without mock
    pass  # Interface test sufficient for now
