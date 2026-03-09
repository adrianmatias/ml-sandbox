#!/usr/bin/env python3
"""Generate testset - thin orchestration script."""

from src.evaluation.testset import TestSet


def main():
    """Generate a testset using the TestSet abstraction."""
    print("🚀 Starting testset generation...")
    testset = TestSet()
    testset.generate()
    print(f"✅ Testset generation complete. Total: {len(testset)} samples")


if __name__ == "__main__":
    main()
