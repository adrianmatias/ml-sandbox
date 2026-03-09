from src.const import CONST
from src.evaluation.testset import TestSet, TestSetItem


def test_testset_item_creation():
    """Test TestSetItem dataclass."""
    item = TestSetItem(
        question="What is the meaning of life?",
        ground_truth="42",
        reference_contexts=["Some context"],
        persona_name="Philosopher",
    )
    assert item.question == "What is the meaning of life?"
    assert item.ground_truth == "42"
    assert len(item.reference_contexts) == 1


def test_testset_class():
    """Test TestSet class functionality."""
    testset = TestSet()
    # We can't easily test generation without long runtime,
    # so test the interface
    assert hasattr(testset, "generate")
    assert hasattr(testset, "load")
    assert hasattr(testset, "to_testset_items")
    assert testset.testset_path == CONST.loc.testset
