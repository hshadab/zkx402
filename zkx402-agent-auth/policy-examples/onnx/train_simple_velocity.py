#!/usr/bin/env python3
"""
Simple Velocity Policy for JOLT Atlas End-to-End Testing

Creates a minimal velocity policy model (5 features) that fits
comfortably within the original JOLT Atlas limits (< 64 elements).

This is used for testing the full JOLT proving pipeline before
moving to larger transformed models.
"""

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


class SimpleVelocityModel(nn.Module):
    """
    Minimal velocity policy model for testing.

    Architecture: 5 → 8 → 2
    Total parameters: ~58 (<< 64 elements, safe for original JOLT Atlas)

    Inputs (5 features):
      1. amount (normalized)
      2. balance (normalized)
      3. velocity_1h (normalized)
      4. velocity_24h (normalized)
      5. vendor_trust (0-1)

    Outputs (2):
      1. approved_score (0-1)
      2. risk_score (0-1)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def generate_training_data(num_samples=1000):
    """Generate synthetic training data for velocity policy."""
    X = []
    y = []

    for _ in range(num_samples):
        # Random features (all normalized 0-1)
        amount = np.random.rand()
        balance = np.random.rand()
        velocity_1h = np.random.rand() * 0.5  # Usually lower than amount
        velocity_24h = np.random.rand() * 0.8
        vendor_trust = np.random.rand()

        features = [amount, balance, velocity_1h, velocity_24h, vendor_trust]

        # Simple policy rules:
        # - Approved if: amount < 0.1 * balance AND velocity_1h < 0.05 * balance AND trust > 0.5
        # - Risk score based on amount/balance ratio

        approved = (
            amount < 0.1 * balance and
            velocity_1h < 0.05 * balance and
            vendor_trust > 0.5
        )

        risk_score = min(1.0, amount / max(balance, 0.1))

        X.append(features)
        y.append([1.0 if approved else 0.0, risk_score])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(model, X, y, epochs=100):
    """Train the model."""
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print(f"\n✓ Training complete! Final loss: {loss.item():.4f}")


def export_to_onnx(model, filename="simple_velocity_policy.onnx"):
    """Export model to ONNX format."""
    dummy_input = torch.randn(1, 5)

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['features'],
        output_names=['scores'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'scores': {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to {filename}")


def test_model(model):
    """Test model with sample inputs."""
    print("\n" + "="*60)
    print("Testing Simple Velocity Model")
    print("="*60)

    test_cases = [
        {
            "name": "✅ Approved: Small amount, good balance",
            "features": [0.05, 0.8, 0.01, 0.05, 0.7],  # amount, balance, vel_1h, vel_24h, trust
            "expected": "APPROVED"
        },
        {
            "name": "❌ Rejected: Amount too high",
            "features": [0.5, 0.8, 0.01, 0.05, 0.7],  # 50% of balance
            "expected": "REJECTED"
        },
        {
            "name": "❌ Rejected: Low vendor trust",
            "features": [0.05, 0.8, 0.01, 0.05, 0.3],  # trust < 0.5
            "expected": "REJECTED"
        },
    ]

    model.eval()
    with torch.no_grad():
        for case in test_cases:
            features = torch.tensor([case["features"]], dtype=torch.float32)
            outputs = model(features)

            approved_score = outputs[0][0].item()
            risk_score = outputs[0][1].item()

            decision = "APPROVED" if approved_score > 0.5 else "REJECTED"

            print(f"\n{case['name']}")
            print(f"  Features: {case['features']}")
            print(f"  Approved score: {approved_score:.3f}")
            print(f"  Risk score: {risk_score:.3f}")
            print(f"  Decision: {decision}")
            print(f"  Expected: {case['expected']} {'✓' if decision == case['expected'] else '✗'}")


def main():
    print("="*60)
    print("Simple Velocity Policy for JOLT Atlas E2E Testing")
    print("="*60)

    # 1. Generate training data
    print("\n[1/4] Generating training data...")
    X_train, y_train = generate_training_data(num_samples=1000)
    print(f"  ✓ Generated {len(X_train)} samples")
    print(f"  ✓ Feature shape: {X_train.shape}")

    # 2. Create model
    print("\n[2/4] Creating model...")
    model = SimpleVelocityModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model architecture: 5 → 8 → 2")
    print(f"  ✓ Total parameters: {total_params}")
    print(f"  ✓ Tensor elements: ~{total_params} (<< 64, safe for JOLT Atlas)")

    # 3. Train model
    print("\n[3/4] Training model...")
    train_model(model, X_train, y_train, epochs=100)

    # 4. Test model
    print("\n[4/4] Testing model...")
    test_model(model)

    # 5. Export to ONNX
    print("\n[5/5] Exporting to ONNX...")
    export_to_onnx(model, "simple_velocity_policy.onnx")

    print("\n" + "="*60)
    print("✅ Simple Model Ready for JOLT Proving!")
    print("="*60)

    print("\nNext step: Run JOLT Atlas proof")
    print("  cd ../../jolt-prover")
    print("  cargo run --release --example simple_velocity_e2e")


if __name__ == "__main__":
    main()
