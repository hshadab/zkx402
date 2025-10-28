#!/usr/bin/env python3
"""
Create demonstration ONNX models showcasing JOLT Atlas enhancements.

This script generates multiple ONNX models that demonstrate:
1. Comparison operations (Greater, Less, GreaterEqual)
2. Tensor operations (Slice, Identity, Reshape)
3. MatMult with 1D tensors
4. Authorization use cases

All models use integer-scaled operations compatible with JOLT Atlas.
"""

import torch
import torch.nn as nn
import onnx
from pathlib import Path

# Scale factor for integer operations (must match JOLT Atlas)
SCALE = 100


class ComparisonDemo(nn.Module):
    """Demonstrates Greater, Less, and GreaterEqual operations."""

    def __init__(self):
        super().__init__()

    def forward(self, amount, balance, threshold):
        """
        Args:
            amount: Transaction amount (scaled by 100)
            balance: Account balance (scaled by 100)
            threshold: Trust threshold (scaled by 100)

        Returns:
            Tuple of (amount_ok, balance_ok, threshold_ok)
        """
        # Greater: balance > amount
        balance_ok = (balance > amount).int()

        # Less: amount < (balance / 2)
        half_balance = balance // 2
        amount_ok = (amount < half_balance).int()

        # GreaterEqual: threshold >= 50
        threshold_ok = (threshold >= 50).int()

        return balance_ok, amount_ok, threshold_ok


class TensorOpsDemo(nn.Module):
    """Demonstrates Slice, Identity, and Reshape operations."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [5] (scaled integers)

        Returns:
            Sliced, reshaped result
        """
        # Identity: pass-through
        x_identity = x

        # Slice: extract first 3 elements
        x_sliced = x[:3]

        # Reshape: [3] -> [3, 1]
        x_reshaped = x_sliced.reshape(3, 1)

        # Flatten back: [3, 1] -> [3]
        x_flat = x_reshaped.flatten()

        return x_flat


class MatMult1DDemo(nn.Module):
    """Demonstrates MatMult with 1D tensor outputs."""

    def __init__(self):
        super().__init__()
        # Weight matrix for [5] x [5, 3] -> [3] (1D output)
        # Note: Use float32 for nn.Parameter, will convert to int in forward
        weight_data = torch.randint(0, 10, (5, 3), dtype=torch.int32).float()
        self.weight = nn.Parameter(weight_data, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Input vector of shape [5] (scaled integers)

        Returns:
            Output vector of shape [3] (1D tensor)
        """
        # Matrix-vector multiplication: [5] x [5, 3] -> [3]
        # Note: PyTorch will produce 1D output
        output = torch.matmul(x.float(), self.weight).int()
        return output


class SimpleAuthPolicy(nn.Module):
    """Simple rule-based authorization policy."""

    def __init__(self):
        super().__init__()

    def forward(self, amount, balance, velocity_1h, velocity_24h, vendor_trust):
        """
        Authorization rules:
        1. amount < 10% of balance
        2. vendor_trust > 50
        3. velocity_1h < 500
        4. velocity_24h < 2000

        Args:
            All inputs scaled by 100

        Returns:
            approved: 1 if all rules pass, 0 otherwise
        """
        # Rule 1: amount < balance * 0.1
        ten_percent = balance // 10
        rule1 = (amount < ten_percent).int()

        # Rule 2: vendor_trust > 50
        rule2 = (vendor_trust > 50).int()

        # Rule 3: velocity_1h < 500
        rule3 = (velocity_1h < 500).int()

        # Rule 4: velocity_24h < 2000
        rule4 = (velocity_24h < 2000).int()

        # All rules must pass
        approved = rule1 * rule2 * rule3 * rule4

        return approved


class NeuralAuthPolicy(nn.Module):
    """Neural network-based authorization policy."""

    def __init__(self):
        super().__init__()
        # Small integer-scaled network
        self.fc1 = nn.Linear(5, 8, bias=True)
        self.fc2 = nn.Linear(8, 4, bias=True)
        self.fc3 = nn.Linear(4, 1, bias=True)

        # Initialize with small integer weights
        with torch.no_grad():
            self.fc1.weight.data = torch.randint(-5, 5, (8, 5), dtype=torch.float32)
            self.fc1.bias.data = torch.randint(-5, 5, (8,), dtype=torch.float32)

            self.fc2.weight.data = torch.randint(-5, 5, (4, 8), dtype=torch.float32)
            self.fc2.bias.data = torch.randint(-5, 5, (4,), dtype=torch.float32)

            self.fc3.weight.data = torch.randint(-5, 5, (1, 4), dtype=torch.float32)
            self.fc3.bias.data = torch.zeros(1, dtype=torch.float32)

    def forward(self, amount, balance, velocity_1h, velocity_24h, vendor_trust):
        """
        Args:
            All inputs scaled by 100 (e.g., 5 means 0.05)

        Returns:
            risk_score: Integer risk score (higher = riskier)
        """
        # Stack inputs into feature vector
        x = torch.stack([amount, balance, velocity_1h, velocity_24h, vendor_trust])

        # Forward pass with ReLU approximated by Clip(0, inf)
        x = self.fc1(x)
        x = torch.clamp(x, min=0)  # ReLU via Clip

        x = self.fc2(x)
        x = torch.clamp(x, min=0)

        x = self.fc3(x)

        # Threshold: score < 50 is approved
        approved = (x < 50).int()

        return approved


def export_model(model, dummy_inputs, filename, opset_version=14):
    """Export PyTorch model to ONNX format."""
    output_path = Path(__file__).parent / filename

    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=False,
        input_names=[f"input_{i}" for i in range(len(dummy_inputs))],
        output_names=["output"],
        dynamic_axes={},
    )

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f"✓ Created {filename}")
    print(f"  Ops: {len(onnx_model.graph.node)}")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    print()

    return output_path


def main():
    """Generate all demonstration ONNX models."""
    print("=" * 60)
    print("Creating JOLT Atlas Demonstration Models")
    print("=" * 60)
    print()

    # 1. Comparison Operations Demo
    print("[1/5] Comparison Operations Demo")
    model = ComparisonDemo()
    dummy_inputs = (
        torch.tensor(500, dtype=torch.int32),  # amount = $5.00
        torch.tensor(10000, dtype=torch.int32),  # balance = $100.00
        torch.tensor(75, dtype=torch.int32),  # threshold = 0.75
    )
    export_model(model, dummy_inputs, "comparison_demo.onnx")

    # 2. Tensor Operations Demo
    print("[2/5] Tensor Operations Demo")
    model = TensorOpsDemo()
    dummy_inputs = (torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32),)
    export_model(model, dummy_inputs, "tensor_ops_demo.onnx")

    # 3. MatMult 1D Demo
    print("[3/5] MatMult 1D Output Demo")
    model = MatMult1DDemo()
    dummy_inputs = (torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),)
    export_model(model, dummy_inputs, "matmult_1d_demo.onnx")

    # 4. Simple Authorization Policy
    print("[4/5] Simple Rule-Based Authorization")
    model = SimpleAuthPolicy()
    dummy_inputs = (
        torch.tensor(500, dtype=torch.int32),  # amount = $5.00
        torch.tensor(10000, dtype=torch.int32),  # balance = $100.00
        torch.tensor(200, dtype=torch.int32),  # velocity_1h = $2.00
        torch.tensor(1500, dtype=torch.int32),  # velocity_24h = $15.00
        torch.tensor(80, dtype=torch.int32),  # vendor_trust = 0.80
    )
    export_model(model, dummy_inputs, "simple_auth.onnx")

    # 5. Neural Authorization Policy
    print("[5/5] Neural Network Authorization")
    model = NeuralAuthPolicy()
    model.eval()
    dummy_inputs = (
        torch.tensor(500.0),  # amount
        torch.tensor(10000.0),  # balance
        torch.tensor(200.0),  # velocity_1h
        torch.tensor(1500.0),  # velocity_24h
        torch.tensor(80.0),  # vendor_trust
    )
    export_model(model, dummy_inputs, "neural_auth.onnx")

    print("=" * 60)
    print("✓ All models created successfully!")
    print()
    print("Models created:")
    print("  1. comparison_demo.onnx     - Greater, Less, GreaterEqual")
    print("  2. tensor_ops_demo.onnx     - Slice, Identity, Reshape")
    print("  3. matmult_1d_demo.onnx     - MatMult with 1D output")
    print("  4. simple_auth.onnx         - Rule-based authorization")
    print("  5. neural_auth.onnx         - Neural network authorization")
    print()
    print("Next steps:")
    print("  1. Test models with: python test_demo_models.py")
    print("  2. Use in JOLT Atlas: cargo run --example <model_name>")
    print("=" * 60)


if __name__ == "__main__":
    main()
