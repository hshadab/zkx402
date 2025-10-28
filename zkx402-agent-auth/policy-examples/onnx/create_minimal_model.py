#!/usr/bin/env python3
"""
Create a minimal ONNX model using ONLY supported JOLT Atlas operations.

Supported operations:
- Add, Sub, Mul (no Div!)
- Greater (>), GreaterEqual (>=), Less (<)
- MatMult
- Clip (for ReLU)
"""

import torch
import torch.nn as nn
import onnx

class MinimalAuth(nn.Module):
    """
    Minimal authorization model using only supported operations.

    Logic: Approve if (balance - amount) > threshold
    """

    def __init__(self):
        super().__init__()

    def forward(self, amount, balance, velocity_1h, velocity_24h, vendor_trust):
        """
        Args:
            amount: Transaction amount (scaled integer)
            balance: Account balance (scaled integer)
            velocity_1h: 1-hour velocity (scaled integer)
            velocity_24h: 24-hour velocity (scaled integer)
            vendor_trust: Vendor trust score (0-100)

        Returns:
            risk score (higher = safer)
        """
        # Simple authorization logic using only Sub operation
        # Risk score = balance - amount - velocity_24h + vendor_trust
        risk_score = balance - amount - velocity_24h + vendor_trust

        return risk_score


def create_minimal_model():
    """Create and export minimal authorization model."""

    model = MinimalAuth()
    model.eval()

    # Example inputs (scaled by 100)
    amount = torch.tensor([500], dtype=torch.int32)        # $5.00
    balance = torch.tensor([10000], dtype=torch.int32)     # $100.00
    velocity_1h = torch.tensor([200], dtype=torch.int32)   # Recent spending
    velocity_24h = torch.tensor([1500], dtype=torch.int32) # Daily spending
    vendor_trust = torch.tensor([80], dtype=torch.int32)   # Trust score

    # Export to ONNX
    torch.onnx.export(
        model,
        (amount, balance, velocity_1h, velocity_24h, vendor_trust),
        "minimal_auth.onnx",
        input_names=["amount", "balance", "velocity_1h", "velocity_24h", "vendor_trust"],
        output_names=["risk_score"],
        opset_version=13,
        do_constant_folding=False,
        export_params=True
    )

    # Verify the model
    onnx_model = onnx.load("minimal_auth.onnx")
    onnx.checker.check_model(onnx_model)

    print("✅ Created minimal_auth.onnx")
    print("\nOperations used:")
    for node in onnx_model.graph.node:
        print(f"  - {node.op_type}")

    # Test the model
    print("\nTest inference:")
    print(f"  Inputs: amount=500, balance=10000, velocity_1h=200, velocity_24h=1500, vendor_trust=80")
    output = model(amount, balance, velocity_1h, velocity_24h, vendor_trust)
    print(f"  Risk Score: {output.item()}")
    print(f"  Calculation: 10000 - 500 - 1500 + 80 = {output.item()}")
    print(f"  Approved: {'YES' if output.item() > 50 else 'NO'} (risk_score > 50)")


if __name__ == "__main__":
    create_minimal_model()
