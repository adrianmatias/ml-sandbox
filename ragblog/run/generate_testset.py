#!/usr/bin/env python3
"""Generate eval_set - thin orchestration script."""

from src.evaluation.eval_set import EvalSet


def main():
    """Generate a eval_set using the EvalSet abstraction."""
    print("🚀 Starting eval_set generation...")
    testset = EvalSet()
    testset.generate()
    print(f"✅ Testset generation complete. Total: {len(testset)} samples")


if __name__ == "__main__":
    main()
