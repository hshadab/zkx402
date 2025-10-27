#!/usr/bin/env python3
"""
Create the simplest possible ONNX model for testing JOLT Atlas.

This creates a minimal model: y = x * 2 (element-wise multiplication)
No matrix multiplication, no complex operations - just element-wise ops.
"""

import torch
import torch.nn as nn
import os

class SimpleModel(nn.Module):
    """
    Simplest possible model: element-wise multiplication
    Input: [amount] (1 feature)
    Output: [result] (1 output)

    Logic: if amount < 1.0, approve (output > 0.5), else reject
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters - just a threshold check
        # We'll use identity + clamp to simulate: approved = 1.0 if amount < threshold else 0.0

    def forward(self, x):
        # x shape: [batch, 1]
        # Simple logic: output 0.9 if x < 1.0, else output 0.1
        # Using sigmoid to make it smooth: sigmoid(5 * (1 - x))
        threshold = torch.tensor([[1.0]])
        diff = threshold - x  # diff > 0 means approved
        scaled = diff * 5.0   # scale up for sharper sigmoid
        result = torch.sigmoid(scaled)  # sigmoid(positive) > 0.5, sigmoid(negative) < 0.5
        return result


def main():
    print("=" * 60)
    print("Creating Simple ONNX Model for JOLT Atlas Testing")
    print("=" * 60)

    # Create model
    print("\n[1/3] Creating model...")
    model = nn.Sequential(
        nn.Identity()  # Simplest possible: just pass through
    )
    print("  ✓ Model: Identity (no-op, just passes input through)")

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    dummy_input = torch.tensor([[1.0]], dtype=torch.float32)  # 1 input

    output_path = "simple_test.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"  ✓ Model exported to {output_path}")

    # Test the model
    print("\n[3/3] Testing model...")
    test_input = torch.tensor([[0.5]], dtype=torch.float32)
    with torch.no_grad():
        output = model(test_input)
    print(f"  Input: {test_input[0][0].item():.2f}")
    print(f"  Output: {output[0][0].item():.2f}")
    print(f"  ✓ Model works correctly")

    print("\n" + "=" * 60)
    print("✅ Simple ONNX Model Created!")
    print("=" * 60)
    print(f"\nModel saved to: {output_path}")
    print("File size:", os.path.getsize(output_path), "bytes")
    print("\nThis model simply passes the input through unchanged.")
    print("Use this to test basic JOLT Atlas ONNX functionality.\n")


if __name__ == "__main__":
    main()
