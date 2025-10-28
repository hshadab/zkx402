#!/usr/bin/env python3
"""
Train a simple velocity-based spending authorization policy using PyTorch.

This creates an ONNX model that can be proven with JOLT Atlas zkML.

Policy: Approve agent spending if:
  - amount < balance * 0.1 (max 10% of balance)
  - velocity_1h < threshold_1h
  - velocity_24h < threshold_24h
  - vendor_trust > minimum_trust

The model learns to classify transactions as authorized/unauthorized based on these rules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class VelocityPolicyModel(nn.Module):
    """
    Simple feedforward network for spending authorization.

    Architecture: 5 inputs → 16 hidden → 8 hidden → 2 outputs

    Inputs:
      1. amount (scaled, micro-USDC / 1M)
      2. balance (scaled, micro-USDC / 1M)
      3. velocity_1h (scaled, micro-USDC / 1M)
      4. velocity_24h (scaled, micro-USDC / 1M)
      5. vendor_trust (0-1)

    Outputs:
      1. approved_score (sigmoid, 0-1)
      2. risk_score (sigmoid, 0-1)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()  # Output probabilities
        )

    def forward(self, x):
        return self.layers(x)


def generate_training_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data based on policy rules.

    Returns:
        X: (n_samples, 5) feature matrix
        y: (n_samples, 2) labels [approved, risk_score]
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Generate random transaction
        balance = np.random.uniform(0.1, 100.0)  # $0.1M to $100M micro-USDC
        amount = np.random.uniform(0.001, balance * 0.5)  # Up to 50% of balance
        velocity_1h = np.random.uniform(0, balance * 0.3)
        velocity_24h = np.random.uniform(velocity_1h, balance * 0.8)
        vendor_trust = np.random.uniform(0, 1)

        # Apply policy rules
        rule_1 = amount < balance * 0.1  # Max 10% of balance
        rule_2 = velocity_1h < balance * 0.05  # Max 5% velocity in 1h
        rule_3 = velocity_24h < balance * 0.2  # Max 20% velocity in 24h
        rule_4 = vendor_trust > 0.5  # Minimum trust threshold

        # Approved if all rules pass
        approved = 1.0 if (rule_1 and rule_2 and rule_3 and rule_4) else 0.0

        # Risk score (higher = riskier)
        risk = 0.0
        if not rule_1:
            risk += 0.4
        if not rule_2:
            risk += 0.3
        if not rule_3:
            risk += 0.2
        if not rule_4:
            risk += 0.1

        X.append([amount, balance, velocity_1h, velocity_24h, vendor_trust])
        y.append([approved, risk])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    """Train the model using supervised learning."""

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print(f"\n✓ Training complete! Final loss: {loss.item():.4f}")


def export_to_onnx(model: nn.Module, filename: str = "velocity_policy.onnx"):
    """Export the trained model to ONNX format for JOLT Atlas."""

    # Create dummy input (5 features)
    dummy_input = torch.randn(1, 5)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['output'],  # Single output with 2 values
        dynamic_axes={
            'features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to {filename}")


def test_model(model: nn.Module):
    """Test the model with sample transactions."""

    print("\n" + "="*60)
    print("Testing Model with Sample Transactions")
    print("="*60)

    test_cases = [
        {
            "name": "✅ Approved: Small amount, low velocity, trusted vendor",
            "features": [0.05, 10.0, 0.02, 0.1, 0.8],  # amount, balance, vel_1h, vel_24h, trust
            "expected": "APPROVED"
        },
        {
            "name": "❌ Rejected: Amount too high (>10% balance)",
            "features": [2.0, 10.0, 0.02, 0.1, 0.8],
            "expected": "REJECTED"
        },
        {
            "name": "❌ Rejected: Velocity too high",
            "features": [0.05, 10.0, 0.8, 2.0, 0.8],
            "expected": "REJECTED"
        },
        {
            "name": "❌ Rejected: Untrusted vendor",
            "features": [0.05, 10.0, 0.02, 0.1, 0.2],
            "expected": "REJECTED"
        },
    ]

    model.eval()
    with torch.no_grad():
        for case in test_cases:
            features = torch.tensor([case["features"]], dtype=torch.float32)
            output = model(features)
            approved_score = output[0][0].item()
            risk_score = output[0][1].item()

            decision = "APPROVED" if approved_score > 0.5 else "REJECTED"

            print(f"\n{case['name']}")
            print(f"  Features: amount={case['features'][0]:.2f}, "
                  f"balance={case['features'][1]:.2f}, "
                  f"vel_1h={case['features'][2]:.2f}, "
                  f"vel_24h={case['features'][3]:.2f}, "
                  f"trust={case['features'][4]:.2f}")
            print(f"  Decision: {decision} (score: {approved_score:.3f})")
            print(f"  Risk: {risk_score:.3f}")
            print(f"  Expected: {case['expected']} {'✓' if decision == case['expected'] else '✗'}")


def main():
    print("="*60)
    print("ZKx402 Agent Authorization - Velocity Policy Training")
    print("="*60)

    # 1. Generate training data
    print("\n[1/5] Generating training data...")
    X_train, y_train = generate_training_data(n_samples=10000)
    print(f"  ✓ Generated {len(X_train)} samples")
    print(f"  ✓ Approval rate: {y_train[:, 0].mean()*100:.1f}%")

    # 2. Create model
    print("\n[2/5] Creating model...")
    model = VelocityPolicyModel()
    print(f"  ✓ Model architecture: 5 → 16 → 8 → 2")
    print(f"  ✓ Total parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Train model
    print("\n[3/5] Training model...")
    train_model(model, X_train, y_train, epochs=100)

    # 4. Test model
    print("\n[4/5] Testing model...")
    test_model(model)

    # 5. Export to ONNX
    print("\n[5/5] Exporting to ONNX...")
    export_to_onnx(model, "velocity_policy.onnx")

    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use velocity_policy.onnx with JOLT Atlas prover")
    print("2. Generate zero-knowledge proofs for authorization")
    print("3. Integrate with zkx402-agent-auth service")
    print("\nExample proof generation:")
    print("  cd ../jolt-prover")
    print("  cargo run --release --example simple_auth")
    print()


if __name__ == "__main__":
    main()
