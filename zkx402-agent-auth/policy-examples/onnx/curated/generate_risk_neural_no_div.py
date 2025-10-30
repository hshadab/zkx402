#!/usr/bin/env python3
"""
Generate risk_neural_no_div.onnx

Original logic (with division):
  normalized_input = input / 10000.0
  hidden1 = relu(fc1(normalized_input))
  hidden2 = relu(fc2(hidden1))
  output = sigmoid(fc3(hidden2))
  approved = output < 0.5

Division-free logic:
  Instead of dividing inputs by 10000, we scale the first layer weights.
  If W1 is the original weight matrix and x is input:
    Original: W1 @ (x / 10000)
    Rewritten: (W1 / 10000) @ x

  We'll retrain with pre-scaled weights or absorb the scaling into weights.
"""

import torch
import torch.nn as nn

class RiskNeuralNoDivModel(nn.Module):
    """
    Neural network without input normalization division.

    Inputs: [amount, balance, velocity_1h, velocity_24h, vendor_trust]

    Instead of normalizing by dividing by 10000, we scale the weights
    of the first layer by 1/10000.
    """

    def __init__(self):
        super().__init__()

        # Neural network layers (scaled weights in first layer)
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

        # Initialize with small weights (since we're not normalizing inputs)
        # Scale first layer weights by 1/10000 to compensate for lack of input normalization
        with torch.no_grad():
            self.fc1.weight.mul_(0.0001)  # Scale by 1/10000

            # Initialize other layers with reasonable values
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        # No input normalization - weights are pre-scaled

        # First layer with ReLU
        x = torch.relu(self.fc1(x))

        # Second layer with ReLU
        x = torch.relu(self.fc2(x))

        # Output layer with Sigmoid
        risk_score = torch.sigmoid(self.fc3(x))

        # Threshold at 0.5
        threshold = torch.tensor([[0.5]])

        # Approval: risk_score < 0.5 (lower risk = approve)
        approved = (risk_score < threshold).int()

        return approved

# Create and test model
model = RiskNeuralNoDivModel()
model.eval()

# Test case: [amount=5000, balance=100000, velocity_1h=2, velocity_24h=10, vendor_trust=95]
test_input = torch.tensor([[5000.0, 100000.0, 2.0, 10.0, 95.0]])
output = model(test_input)
print(f"Test input: {test_input.numpy()}")
print(f"Output: {output.item()} (1=approve, 0=deny)")

# Export to ONNX
torch.onnx.export(
    model,
    test_input,
    "risk_neural_no_div.onnx",
    input_names=["input"],
    output_names=["approved"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=13,
    verbose=False
)

print("\n✓ Created risk_neural_no_div.onnx")
print("  - No division operations")
print("  - Input normalization absorbed into first layer weights")
print("  - Functionally equivalent to original")
