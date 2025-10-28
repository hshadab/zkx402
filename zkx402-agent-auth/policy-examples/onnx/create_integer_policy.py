#!/usr/bin/env python3
"""
Create INTEGER-ONLY ONNX authorization policy for JOLT Atlas.

CRITICAL: JOLT Atlas requires i32 tensors, NOT f32.
This version uses ONLY integer operations by scaling inputs by 100.

Example:
  - $0.05 → 5 (scaled by 100)
  - $10.00 → 1000
  - Trust 0.80 → 80

Policy Rules (all must pass):
1. Amount < 10% of balance
2. Velocity 1h < 5% of balance
3. Velocity 24h < 20% of balance
4. Vendor trust > 50 (0.5 scaled by 100)

Inputs: [amount_scaled, balance_scaled, velocity_1h_scaled, velocity_24h_scaled, vendor_trust_scaled]
Output: [approved_score_scaled] (0 = reject, 100 = approve)
"""

import torch
import torch.nn as nn
import os


class IntegerRuleBasedAuth(nn.Module):
    """
    INTEGER-ONLY rule-based authorization for JOLT Atlas.

    All inputs are scaled integers (multiply by 100).
    NO MatMult, NO Linear layers, NO learnable parameters.
    Only uses: Add, Sub, Mul, Div, ReLU, GE/LE comparisons
    """

    def forward(self, x):
        # x shape: [batch, 5] = [amount, balance, velocity_1h, velocity_24h, vendor_trust]
        # All values are INTEGERS (scaled by 100)

        amount = x[:, 0:1]        # [batch, 1]
        balance = x[:, 1:2]       # [batch, 1]
        velocity_1h = x[:, 2:3]   # [batch, 1]
        velocity_24h = x[:, 3:4]  # [batch, 1]
        vendor_trust = x[:, 4:5]  # [batch, 1]

        # Rule 1: amount < balance * 10 / 100  (10% of balance)
        # Compute: balance * 10 / 100 - amount
        # If positive, rule passes
        balance_10pct = balance * 10 // 100  # Integer division
        rule1_margin = balance_10pct - amount
        rule1_pass = torch.relu(rule1_margin)  # Positive if passes, 0 if fails

        # Rule 2: velocity_1h < balance * 5 / 100  (5% of balance)
        balance_5pct = balance * 5 // 100
        rule2_margin = balance_5pct - velocity_1h
        rule2_pass = torch.relu(rule2_margin)

        # Rule 3: velocity_24h < balance * 20 / 100  (20% of balance)
        balance_20pct = balance * 20 // 100
        rule3_margin = balance_20pct - velocity_24h
        rule3_pass = torch.relu(rule3_margin)

        # Rule 4: vendor_trust > 50  (0.5 scaled by 100)
        # Compute: vendor_trust - 50
        rule4_margin = vendor_trust - 50
        rule4_pass = torch.relu(rule4_margin)

        # Convert margins to binary using ONLY comparisons (no clamp)
        # If margin > 0, rule passes (output 1), else fails (output 0)
        # We sum all positive margins to count how many rules passed
        total_pass_count = (
            (rule1_pass > 0).to(torch.int32) +
            (rule2_pass > 0).to(torch.int32) +
            (rule3_pass > 0).to(torch.int32) +
            (rule4_pass > 0).to(torch.int32)
        )

        # All 4 rules must pass → total_pass_count == 4
        # Output 100 if all pass, 0 otherwise
        # Using comparison: if total_pass_count >= 4, output 100
        approved_score = ((total_pass_count >= 4).to(torch.int32)) * 100

        return approved_score


def test_policy():
    """Test the policy with various scenarios"""
    model = IntegerRuleBasedAuth()
    model.eval()

    test_cases = [
        # [amount, balance, velocity_1h, velocity_24h, vendor_trust, expected]
        # All values scaled by 100 (so $0.05 = 5, $10 = 1000, trust 0.8 = 80)
        ("Approved: Small tx, low velocity, trusted vendor",
         [5, 1000, 2, 10, 80], True),

        ("Rejected: Amount too large (>10% of balance)",
         [200, 1000, 2, 10, 80], False),

        ("Rejected: Hourly velocity exceeded",
         [5, 1000, 60, 10, 80], False),

        ("Rejected: Daily velocity exceeded",
         [5, 1000, 2, 250, 80], False),

        ("Rejected: Vendor not trusted",
         [5, 1000, 2, 10, 30], False),

        ("Edge case: Exactly at threshold",
         [100, 1000, 50, 200, 50], True),
    ]

    print("=" * 70)
    print("Testing INTEGER-ONLY Rule-Based Authorization Policy")
    print("=" * 70)

    for description, inputs, should_approve in test_cases:
        # Use int32 tensors (critical for JOLT Atlas)
        input_tensor = torch.tensor([inputs], dtype=torch.int32)

        with torch.no_grad():
            output = model(input_tensor)
            approved_score = output.item()
            approved = approved_score > 50  # Threshold: 50/100

        status = "✓ PASS" if (approved == should_approve) else "✗ FAIL"
        print(f"\n{status} {description}")
        print(f"  Inputs: {inputs}")
        print(f"  Score: {approved_score} -> {'APPROVED' if approved else 'REJECTED'}")
        print(f"  Expected: {'APPROVED' if should_approve else 'REJECTED'}")


def main():
    print("=" * 70)
    print("Creating INTEGER-ONLY ONNX Authorization Policy for JOLT Atlas")
    print("(All values scaled by 100, using i32 tensors)")
    print("=" * 70)

    # Create model
    print("\n[1/3] Creating model...")
    model = IntegerRuleBasedAuth()
    model.eval()
    print("  ✓ Model: INTEGER-ONLY rule-based (no learnable parameters)")
    print("  ✓ Operations: Add, Sub, Mul, Div, ReLU, Clamp only")
    print("  ✓ NO MatMult, NO Linear layers, NO Sigmoid")
    print("  ✓ Tensor type: int32 (JOLT Atlas compatible)")

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    # Example: $0.05, $10, $0.02, $0.10, trust=0.80 → [5, 1000, 2, 10, 80]
    dummy_input = torch.tensor([[5, 1000, 2, 10, 80]], dtype=torch.int32)

    output_path = "integer_rule_based_auth.onnx"
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
    print("✅ INTEGER-ONLY ONNX Policy Created!")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")
    print("\nScaling Convention:")
    print("  - Dollar amounts: multiply by 100 (e.g., $0.05 → 5, $10 → 1000)")
    print("  - Trust scores: multiply by 100 (e.g., 0.80 → 80)")
    print("  - Output: 0-100 (0 = reject, 100 = approve)")
    print("\nThis model is JOLT Atlas compatible:")
    print("  ✓ Uses i32 tensors (not f32)")
    print("  ✓ No floating-point operations")
    print("  ✓ No MatMult operations")
    print("\nNext: Test with JOLT Atlas E2E proof generation!")


if __name__ == "__main__":
    main()
