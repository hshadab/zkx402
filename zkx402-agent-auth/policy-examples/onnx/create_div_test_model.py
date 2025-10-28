#!/usr/bin/env python3
"""
Create ONNX model to test Div operation.
Simple model: amount / 100 (convert cents to dollars)
"""

import torch
import torch.nn as nn
import onnx

class DivTest(nn.Module):
    """Simple division test model."""

    def __init__(self):
        super().__init__()

    def forward(self, amount, divisor):
        """
        Args:
            amount: Integer amount in cents
            divisor: Divisor (e.g., 100)

        Returns:
            result: amount / divisor
        """
        return amount / divisor


def create_model():
    """Create and export model."""
    model = DivTest()
    model.eval()

    # Test inputs: 50000 cents / 100 = 500 dollars
    amount = torch.tensor([50000], dtype=torch.int32)
    divisor = torch.tensor([100], dtype=torch.int32)

    # Export to ONNX
    torch.onnx.export(
        model,
        (amount, divisor),
        "div_test.onnx",
        input_names=["amount", "divisor"],
        output_names=["result"],
        opset_version=13,
        do_constant_folding=False,
        export_params=True
    )

    # Verify
    onnx_model = onnx.load("div_test.onnx")
    onnx.checker.check_model(onnx_model)

    print("✅ Created div_test.onnx")
    print("\nInputs:")
    for inp in onnx_model.graph.input:
        print(f"  - {inp.name}: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print("\nOutput:")
    for out in onnx_model.graph.output:
        print(f"  - {out.name}: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    print("\nOperations used:")
    for node in onnx_model.graph.node:
        print(f"  - {node.op_type}")

    # Test
    print("\nTest:")
    print(f"  50000 / 100 = {model(amount, divisor).item()}")


if __name__ == "__main__":
    create_model()
