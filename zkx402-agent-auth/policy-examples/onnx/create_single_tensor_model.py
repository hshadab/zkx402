#!/usr/bin/env python3
"""
Create ONNX model that accepts a SINGLE tensor input of shape [1, 5]
to match how JOLT Atlas passes inputs.
"""

import torch
import torch.nn as nn
import onnx

class SingleTensorAuth(nn.Module):
    """Authorization model with single tensor input [1, 5]."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape [1, 5] containing:
                [amount, balance, velocity_1h, velocity_24h, vendor_trust]

        Returns:
            risk_score: Single value (higher = safer)
        """
        # Extract values from input tensor
        # inputs shape: [1, 5]
        # Flatten to [5]
        flat = inputs.flatten()

        # Extract individual values
        amount = flat[0]
        balance = flat[1]
        velocity_1h = flat[2]
        velocity_24h = flat[3]
        vendor_trust = flat[4]

        # Calculate risk score using only supported operations (Add, Sub, Mul)
        # Risk = balance - amount - velocity_24h + vendor_trust
        risk = balance - amount
        risk = risk - velocity_24h
        risk = risk + vendor_trust

        return risk.unsqueeze(0)  # Return shape [1]


def create_model():
    """Create and export model."""
    model = SingleTensorAuth()
    model.eval()

    # Single tensor input: shape [1, 5]
    inputs = torch.tensor([[500, 10000, 200, 1500, 80]], dtype=torch.int32)

    # Export to ONNX
    torch.onnx.export(
        model,
        inputs,
        "single_tensor_auth.onnx",
        input_names=["inputs"],
        output_names=["risk_score"],
        opset_version=13,
        do_constant_folding=False,
        export_params=True
    )

    # Verify
    onnx_model = onnx.load("single_tensor_auth.onnx")
    onnx.checker.check_model(onnx_model)

    print("✅ Created single_tensor_auth.onnx")
    print("\nInput shape:", [[d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]])
    print("Output shape:", [[d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]])

    print("\nOperations used:")
    for node in onnx_model.graph.node:
        print(f"  - {node.op_type}")

    # Test
    print("\nTest:")
    print(f"  Input: [500, 10000, 200, 1500, 80]")
    output = model(inputs)
    print(f"  Risk Score: {output.item()}")
    print(f"  Calculation: 10000 - 500 - 1500 + 80 = {output.item()}")


if __name__ == "__main__":
    create_model()
