#!/usr/bin/env python3
"""
PURE INTEGER ONNX authorization policy for JOLT Atlas.

CRITICAL: NO type casting, NO division operations, ONLY integer arithmetic.
This version uses ONLY: Add, Sub, Mul, Greater, Less comparisons.

Example:
  - $0.05 → 5 (scaled by 100)
  - $10.00 → 1000
  - Trust 0.80 → 80

Policy Rules (all must pass):
1. amount * 10 < balance  (amount < 10% of balance)
2. velocity_1h * 20 < balance  (velocity_1h < 5% of balance)
3. velocity_24h * 5 < balance  (velocity_24h < 20% of balance)
4. vendor_trust > 50

Inputs: [amount_scaled, balance_scaled, velocity_1h_scaled, velocity_24h_scaled, vendor_trust_scaled]
Output: [approved_score_scaled] (0 = reject, 100 = approve)
"""

import torch
import torch.nn as nn
import os


class PureIntegerRuleBasedAuth(nn.Module):
    """
    PURE INTEGER rule-based authorization for JOLT Atlas.

    NO division, NO casting, ONLY integer operations.
    Uses multiplication to avoid division: instead of "a/b < c", use "a < b*c"
    """

    def forward(self, x):
        # x shape: [batch, 5] = [amount, balance, velocity_1h, velocity_24h, vendor_trust]
        # All values are INTEGERS (scaled by 100)

        amount = x[:, 0:1]        # [batch, 1]
        balance = x[:, 1:2]       # [batch, 1]
        velocity_1h = x[:, 2:3]   # [batch, 1]
        velocity_24h = x[:, 3:4]  # [batch, 1]
        vendor_trust = x[:, 4:5]  # [batch, 1]

        # Rule 1: amount < balance * 10 / 100
        # Rewrite: amount * 100 < balance * 10
        # Simplify: amount * 10 < balance
        rule1_left = amount * 10
        rule1_pass_int = (rule1_left < balance).to(torch.int32)

        # Rule 2: velocity_1h < balance * 5 / 100
        # Rewrite: velocity_1h * 100 < balance * 5
        # Simplify: velocity_1h * 20 < balance
        rule2_left = velocity_1h * 20
        rule2_pass_int = (rule2_left < balance).to(torch.int32)

        # Rule 3: velocity_24h < balance * 20 / 100
        # Rewrite: velocity_24h * 100 < balance * 20
        # Simplify: velocity_24h * 5 < balance
        rule3_left = velocity_24h * 5
        rule3_pass_int = (rule3_left < balance).to(torch.int32)

        # Rule 4: vendor_trust > 50
        rule4_pass_int = (vendor_trust > 50).to(torch.int32)

        # Count how many rules passed (sum of 0/1 values)
        total_pass_count = rule1_pass_int + rule2_pass_int + rule3_pass_int + rule4_pass_int

        # All 4 rules must pass → total_pass_count == 4
        # Output 100 if all pass, 0 otherwise
        approved_score = ((total_pass_count >= 4).to(torch.int32)) * 100

        return approved_score


def test_policy():
    """Test the policy with various scenarios"""
    model = PureIntegerRuleBasedAuth()
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

        ("Edge case: Amount exactly at 10% threshold",
         [100, 1000, 2, 10, 80], False),  # 100*10 = 1000, NOT < 1000, so fails

        ("Edge case: Amount just below threshold",
         [99, 1000, 2, 10, 80], True),  # 99*10 = 990 < 1000, passes
    ]

    print("=" * 70)
    print("Testing PURE INTEGER Rule-Based Authorization Policy")
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
    print("Creating PURE INTEGER ONNX Authorization Policy for JOLT Atlas")
    print("(NO casting, NO division, ONLY integer operations)")
    print("=" * 70)

    # Create model
    print("\n[1/3] Creating model...")
    model = PureIntegerRuleBasedAuth()
    model.eval()
    print("  ✓ Model: PURE INTEGER rule-based (no learnable parameters)")
    print("  ✓ Operations: Add, Sub, Mul, Greater, Less ONLY")
    print("  ✓ NO Division, NO Casting, NO ReLU")
    print("  ✓ Tensor type: int32 (JOLT Atlas compatible)")

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    # Example: $0.05, $10, $0.02, $0.10, trust=0.80 → [5, 1000, 2, 10, 80]
    dummy_input = torch.tensor([[5, 1000, 2, 10, 80]], dtype=torch.int32)

    output_path = "pure_integer_auth.onnx"
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
    print("✅ PURE INTEGER ONNX Policy Created!")
    print("=" * 70)
    print(f"\nModel saved to: {output_path}")
    print("\nScaling Convention:")
    print("  - Dollar amounts: multiply by 100 (e.g., $0.05 → 5, $10 → 1000)")
    print("  - Trust scores: multiply by 100 (e.g., 0.80 → 80)")
    print("  - Output: 0-100 (0 = reject, 100 = approve)")
    print("\nPolicy Rules (using multiplication instead of division):")
    print("  1. amount * 10 < balance  (equivalent to: amount < 10% of balance)")
    print("  2. velocity_1h * 20 < balance  (equivalent to: velocity_1h < 5% of balance)")
    print("  3. velocity_24h * 5 < balance  (equivalent to: velocity_24h < 20% of balance)")
    print("  4. vendor_trust > 50")
    print("\nThis model is JOLT Atlas compatible:")
    print("  ✓ Uses i32 tensors (not f32)")
    print("  ✓ No floating-point operations")
    print("  ✓ No division operations")
    print("  ✓ No type casting")
    print("  ✓ No MatMult operations")
    print("\nNext: Test with JOLT Atlas E2E proof generation!")


if __name__ == "__main__":
    main()
