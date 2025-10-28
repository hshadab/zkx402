#!/usr/bin/env python3
"""
Generate risk_neural.onnx with correct dimensions.
Simpler version without neural network complexity - uses weighted scoring instead.
"""

import torch
import torch.nn as nn
import onnx

class RiskScoring(nn.Module):
    """
    Lightweight risk scoring (simplified from neural network).
    Implements a weighted scoring system using only supported operations.
    """
    def forward(self, amount, balance, velocity_1h, velocity_24h, vendor_trust):
        # Scale inputs to reasonable ranges
        # Balance safety: (balance - amount) / 1000
        balance_score = (balance - amount) // 1000

        # Velocity safety: Check if within reasonable limits
        # Lower velocity = safer (inverse scoring)
        velocity_score_1h = torch.clamp(100 - (velocity_1h // 100), 0, 100)
        velocity_score_24h = torch.clamp(100 - (velocity_24h // 500), 0, 100)

        # Vendor trust is already 0-100
        trust_score = vendor_trust

        # Composite risk score (weighted average)
        # balance_score (40%) + velocity (30%) + trust (30%)
        risk_score = (
            (balance_score * 40) +
            (velocity_score_1h * 15) +
            (velocity_score_24h * 15) +
            (trust_score * 30)
        ) // 100

        # Clamp to 0-100 range
        risk_score = torch.clamp(risk_score, 0, 100)

        # Approve if risk score > 50
        approved = (risk_score > 50).to(torch.int32)

        return risk_score.to(torch.int32), approved

model = RiskScoring()
model.eval()

# Test inputs
amount = torch.tensor([5000], dtype=torch.int32)
balance = torch.tensor([100000], dtype=torch.int32)
velocity_1h = torch.tensor([5000], dtype=torch.int32)
velocity_24h = torch.tensor([20000], dtype=torch.int32)
vendor_trust = torch.tensor([75], dtype=torch.int32)

torch.onnx.export(
    model,
    (amount, balance, velocity_1h, velocity_24h, vendor_trust),
    "curated/risk_neural.onnx",
    input_names=["amount", "balance", "velocity_1h", "velocity_24h", "vendor_trust"],
    output_names=["risk_score", "approved"],
    opset_version=13,
    do_constant_folding=False
)

risk_score, approved = model(amount, balance, velocity_1h, velocity_24h, vendor_trust)
print("✅ 10/10: risk_neural.onnx (weighted scoring)")
print(f"   Test: risk_score={risk_score.item()}, approved={approved.item()}")

# Verify model
onnx_model = onnx.load("curated/risk_neural.onnx")
onnx.checker.check_model(onnx_model)
print("\n✅ Model validated!")

print("\nOperations used:")
for node in onnx_model.graph.node:
    print(f"  - {node.op_type}")
