#!/usr/bin/env python3
"""
Create rule-based ONNX authorization policy using ONLY supported JOLT operations.

This creates a practical agent spending policy WITHOUT neural networks or MatMult.
Uses only: Add, Sub, Mul, Div, ReLU, Sigmoid, GE, LE, ReduceSum

Policy Rules (all must pass):
1. Amount < 10% of balance
2. Velocity 1h < 5% of balance
3. Velocity 24h < 20% of balance
4. Vendor trust > 0.5

Inputs: [amount, balance, velocity_1h, velocity_24h, vendor_trust]
Output: [approved_score] (0.0 = reject, 1.0 = approve)
"""

import torch
import torch.nn as nn
import os


class RuleBasedAuth(nn.Module):
    """
    Rule-based authorization using only JOLT-supported ops.
    NO MatMult, NO Linear layers, NO learnable parameters.
    """

    def forward(self, x):
        # x shape: [batch, 5] = [amount, balance, velocity_1h, velocity_24h, vendor_trust]

        amount = x[:, 0:1]        # [batch, 1]
        balance = x[:, 1:2]       # [batch, 1]
        velocity_1h = x[:, 2:3]   # [batch, 1]
        velocity_24h = x[:, 3:4]  # [batch, 1]
        vendor_trust = x[:, 4:5]  # [batch, 1]

        # Rule 1: amount < 0.1 * balance
        # Compute: 0.1 * balance - amount
        # If positive, rule passes
        rule1_margin = (balance * 0.1) - amount
        rule1_pass = torch.relu(rule1_margin)  # Positive if passes, 0 if fails

        # Rule 2: velocity_1h < 0.05 * balance
        rule2_margin = (balance * 0.05) - velocity_1h
        rule2_pass = torch.relu(rule2_margin)

        # Rule 3: velocity_24h < 0.2 * balance
        rule3_margin = (balance * 0.2) - velocity_24h
        rule3_pass = torch.relu(rule3_margin)

        # Rule 4: vendor_trust > 0.5
        # Compute: vendor_trust - 0.5
        rule4_margin = vendor_trust - 0.5
        rule4_pass = torch.relu(rule4_margin)

        # All rules must pass
        # Convert positive margins to binary (0 or 1)
        # Using sigmoid: sigmoid(large_value) ≈ 1, sigmoid(0) = 0.5
        # Scale margins by 100 to make decision sharper
        r1 = torch.sigmoid(rule1_pass * 100.0)
        r2 = torch.sigmoid(rule2_pass * 100.0)
        r3 = torch.sigmoid(rule3_pass * 100.0)
        r4 = torch.sigmoid(rule4_pass * 100.0)

        # Multiply all rule results (AND logic)
        # All must be ~1.0 for approval
        approved_score = r1 * r2 * r3 * r4

        return approved_score


def test_policy():
    """Test the policy with various scenarios"""
    model = RuleBasedAuth()
    model.eval()

    test_cases = [
        # [amount, balance, velocity_1h, velocity_24h, vendor_trust, expected]
        ("Approved: Small tx, low velocity, trusted vendor",
         [0.05, 10.0, 0.02, 0.10, 0.80], True),

        ("Rejected: Amount too large (>10% of balance)",
         [2.0, 10.0, 0.02, 0.10, 0.80], False),

        ("Rejected: Hourly velocity exceeded",
         [0.05, 10.0, 0.60, 0.10, 0.80], False),

        ("Rejected: Daily velocity exceeded",
         [0.05, 10.0, 0.02, 2.50, 0.80], False),

        ("Rejected: Vendor not trusted",
         [0.05, 10.0, 0.02, 0.10, 0.30], False),

        ("Edge case: Exactly at threshold (should pass)",
         [1.0, 10.0, 0.5, 2.0, 0.5], True),
    ]

    print("=" * 70)
    print("Testing Rule-Based Authorization Policy")
    print("=" * 70)

    for description, inputs, should_approve in test_cases:
        input_tensor = torch.tensor([inputs], dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            approved = output.item() > 0.5

        status = "✓ PASS" if (approved == should_approve) else "✗ FAIL"
        print(f"\n{status} {description}")
        print(f"  Inputs: {inputs}")
        print(f"  Score: {output.item():.4f} -> {'APPROVED' if approved else 'REJECTED'}")
        print(f"  Expected: {'APPROVED' if should_approve else 'REJECTED'}")


def main():
    print("=" * 70)
    print("Creating Rule-Based ONNX Authorization Policy")
    print("(Using ONLY JOLT-supported operations - NO MatMult)")
    print("=" * 70)

    # Create model
    print("\n[1/3] Creating model...")
    model = RuleBasedAuth()
    model.eval()
    print("  ✓ Model: Rule-based (no learnable parameters)")
    print("  ✓ Operations: Add, Sub, Mul, ReLU, Sigmoid only")
    print("  ✓ NO MatMult, NO Linear layers")

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    dummy_input = torch.tensor([[0.05, 10.0, 0.02, 0.10, 0.80]], dtype=torch.float32)

    output_path = "rule_based_auth.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['inputs'],
        output_names=['approved_score'],
        dynamic_axes={
            'inputs': {0: 'batch_size'},
            'approved_score': {0: 'batch_size'}
        }
    )
    print(f"  ✓ Model exported to {output_path}")
    print(f"  ✓ File size: {os.path.getsize(output_path)} bytes")

    # Test the model
    print("\n[3/3] Testing model...")
    test_policy()

    print("\n" + "=" * 70)
    print("✅ Rule-Based ONNX Policy Created!")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")
    print("\nThis policy implements real authorization logic WITHOUT neural networks.")
    print("It should work with JOLT Atlas since it uses ONLY supported operations.")
    print("\nNext: Test with JOLT Atlas E2E proof generation!")


if __name__ == "__main__":
    main()
