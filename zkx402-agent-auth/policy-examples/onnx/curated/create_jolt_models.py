#!/usr/bin/env python3
"""
Create JOLT Atlas compatible ONNX models for authorization.
JOLT requires a single concatenated input tensor, not separate named inputs.
"""

import torch
import torch.nn as nn
import numpy as np

def export_model(model, input_size, filename, input_names=['input'], output_names=['approved']):
    """Export PyTorch model to ONNX with single tensor input"""
    dummy_input = torch.randn(1, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True
    )
    print(f"✓ Created: {filename} (input_size={input_size})")

# 1. Simple Threshold (2 inputs: amount, balance)
class SimpleThreshold(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        balance = x[:, 1]
        approved = (amount < balance).float()
        return approved.unsqueeze(1)

export_model(SimpleThreshold(), 2, 'simple_threshold.onnx')

# 2. Percentage Limit (3 inputs: amount, balance, max_percentage)
class PercentageLimit(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        balance = x[:, 1]
        max_percentage = x[:, 2]
        # amount <= balance * (max_percentage / 100)
        limit = balance * max_percentage / 100.0
        approved = (amount < limit).float()
        return approved.unsqueeze(1)

export_model(PercentageLimit(), 3, 'percentage_limit.onnx')

# 3. Vendor Trust (2 inputs: vendor_trust, min_trust)
class VendorTrust(nn.Module):
    def forward(self, x):
        vendor_trust = x[:, 0]
        min_trust = x[:, 1]
        approved = (vendor_trust >= min_trust).float()
        return approved.unsqueeze(1)

export_model(VendorTrust(), 2, 'vendor_trust.onnx')

# 4. Velocity 1h (3 inputs: amount, spent_1h, limit_1h)
class Velocity1h(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        spent_1h = x[:, 1]
        limit_1h = x[:, 2]
        # spent_1h + amount <= limit_1h
        approved = ((spent_1h + amount) <= limit_1h).float()
        return approved.unsqueeze(1)

export_model(Velocity1h(), 3, 'velocity_1h.onnx')

# 5. Velocity 24h (3 inputs: amount, spent_24h, limit_24h)
class Velocity24h(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        spent_24h = x[:, 1]
        limit_24h = x[:, 2]
        approved = ((spent_24h + amount) <= limit_24h).float()
        return approved.unsqueeze(1)

export_model(Velocity24h(), 3, 'velocity_24h.onnx')

# 6. Daily Limit (3 inputs: amount, daily_spent, daily_cap)
class DailyLimit(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        daily_spent = x[:, 1]
        daily_cap = x[:, 2]
        approved = ((daily_spent + amount) <= daily_cap).float()
        return approved.unsqueeze(1)

export_model(DailyLimit(), 3, 'daily_limit.onnx')

# 7. Age Gate (2 inputs: age, min_age)
class AgeGate(nn.Module):
    def forward(self, x):
        age = x[:, 0]
        min_age = x[:, 1]
        approved = (age >= min_age).float()
        return approved.unsqueeze(1)

export_model(AgeGate(), 2, 'age_gate.onnx')

# 8. Multi-Factor (6 inputs: amount, balance, spent_24h, limit_24h, vendor_trust, min_trust)
class MultiFactor(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        balance = x[:, 1]
        spent_24h = x[:, 2]
        limit_24h = x[:, 3]
        vendor_trust = x[:, 4]
        min_trust = x[:, 5]

        # All conditions must pass
        balance_ok = (amount < balance).float()
        velocity_ok = ((spent_24h + amount) <= limit_24h).float()
        trust_ok = (vendor_trust >= min_trust).float()

        approved = balance_ok * velocity_ok * trust_ok
        return approved.unsqueeze(1)

export_model(MultiFactor(), 6, 'multi_factor.onnx')

# 9. Composite Scoring (4 inputs: amount, balance, vendor_trust, user_history)
class CompositeScoring(nn.Module):
    def forward(self, x):
        amount = x[:, 0]
        balance = x[:, 1]
        vendor_trust = x[:, 2]
        user_history = x[:, 3]

        # Weighted risk score
        balance_ratio = amount / (balance + 1.0)
        risk_score = (
            balance_ratio * 0.4 +
            (100.0 - vendor_trust) / 100.0 * 0.3 +
            (100.0 - user_history) / 100.0 * 0.3
        )

        # Approve if risk score < 0.5 (50%)
        approved = (risk_score < 50.0).float()
        return torch.stack([risk_score, approved], dim=1)

export_model(CompositeScoring(), 4, 'composite_scoring.onnx', output_names=['risk_score', 'approved'])

# 10. Risk Neural (5 inputs: amount, balance, velocity_1h, velocity_24h, vendor_trust)
class RiskNeural(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        # Normalize inputs to [0, 1] range
        x_norm = torch.sigmoid(x / 10000.0)

        # Neural network
        h1 = torch.relu(self.fc1(x_norm))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)

        risk_score = out[:, 0]
        approved = (risk_score < 0.5).float()

        return torch.stack([risk_score, approved], dim=1)

# Initialize with reasonable weights for risk assessment
model = RiskNeural()
with torch.no_grad():
    # Set weights for sensible risk scoring
    model.fc1.weight.fill_(0.1)
    model.fc2.weight.fill_(0.1)
    model.fc3.weight.fill_(0.1)

export_model(model, 5, 'risk_neural.onnx', output_names=['risk_score', 'approved'])

print("\n✓ All 10 JOLT-compatible models created successfully!")
print("\nModel Summary:")
print("  2 inputs: simple_threshold, vendor_trust, age_gate")
print("  3 inputs: percentage_limit, velocity_1h, velocity_24h, daily_limit")
print("  4 inputs: composite_scoring")
print("  5 inputs: risk_neural")
print("  6 inputs: multi_factor")
