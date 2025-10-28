#!/usr/bin/env python3
"""Test that all demonstration ONNX models are valid and loadable."""

import onnx
from pathlib import Path

def test_model(model_path):
    """Load and validate an ONNX model."""
    print(f"Testing {model_path.name}...")

    try:
        # Load model
        model = onnx.load(model_path)

        # Check model
        onnx.checker.check_model(model)

        # Print model info
        print(f"  ✓ Valid ONNX model")
        print(f"  Inputs: {len(model.graph.input)}")
        print(f"  Outputs: {len(model.graph.output)}")
        print(f"  Operations: {len(model.graph.node)}")

        # Print operation types
        op_types = set(node.op_type for node in model.graph.node)
        print(f"  Op types: {', '.join(sorted(op_types))}")
        print()

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print()
        return False

def main():
    """Test all demonstration models."""
    models_dir = Path(__file__).parent
    models = [
        "comparison_demo.onnx",
        "tensor_ops_demo.onnx",
        "matmult_1d_demo.onnx",
        "simple_auth.onnx",
        "neural_auth.onnx",
    ]

    print("=" * 60)
    print("Testing JOLT Atlas Demonstration Models")
    print("=" * 60)
    print()

    results = []
    for model_name in models:
        model_path = models_dir / model_name
        if model_path.exists():
            results.append(test_model(model_path))
        else:
            print(f"✗ Model not found: {model_name}")
            results.append(False)

    print("=" * 60)
    if all(results):
        print("✓ All models valid!")
    else:
        print(f"✗ {results.count(False)} models failed")
    print("=" * 60)

if __name__ == "__main__":
    main()
