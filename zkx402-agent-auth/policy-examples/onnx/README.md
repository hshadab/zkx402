# ONNX Policy Models for JOLT Atlas

This directory contains example ONNX models for agent authorization policies that can be proven with JOLT Atlas zkML.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Velocity Policy Model

```bash
python train_velocity.py
```

**Output**: `velocity_policy.onnx`

### 3. Verify ONNX Model

```bash
python -c "import onnx; model = onnx.load('velocity_policy.onnx'); onnx.checker.check_model(model); print('✓ ONNX model is valid')"
```

## Velocity Policy Model

### Architecture

```
Input (5 features)
    ↓
Linear(5 → 16)
    ↓
ReLU
    ↓
Linear(16 → 8)
    ↓
ReLU
    ↓
Linear(8 → 2)
    ↓
Sigmoid
    ↓
Output: [approved_score, risk_score]
```

### Inputs

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `amount` | float32 | 0.0-100.0 | Transaction amount (scaled: micro-USDC / 1M) |
| `balance` | float32 | 0.0-100.0 | Current balance (scaled: micro-USDC / 1M) |
| `velocity_1h` | float32 | 0.0-100.0 | Spending in last 1 hour (scaled) |
| `velocity_24h` | float32 | 0.0-100.0 | Spending in last 24 hours (scaled) |
| `vendor_trust` | float32 | 0.0-1.0 | Vendor trust score |

### Outputs

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `approved_score` | float32 | 0.0-1.0 | Probability of approval (>0.5 = approved) |
| `risk_score` | float32 | 0.0-1.0 | Transaction risk level (0 = safe, 1 = risky) |

### Policy Rules (Learned)

The model is trained to approve transactions when:

1. **Amount < 10% of balance** (anti-overspend)
2. **1h velocity < 5% of balance** (anti-spam)
3. **24h velocity < 20% of balance** (daily limit)
4. **Vendor trust > 0.5** (whitelist-ish)

## Example Usage

### Python Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("velocity_policy.onnx")

# Prepare input (example: $0.05M amount, $10M balance, low velocity, trusted vendor)
features = np.array([[0.05, 10.0, 0.02, 0.1, 0.8]], dtype=np.float32)

# Run inference
outputs = session.run(None, {"features": features})
approved_score = outputs[0][0][0]
risk_score = outputs[0][0][1]

print(f"Approved: {approved_score > 0.5}")
print(f"Score: {approved_score:.3f}")
print(f"Risk: {risk_score:.3f}")
```

### JOLT Atlas Proof Generation

See `../../jolt-prover/examples/simple_auth.rs` for Rust code that:
1. Loads this ONNX model
2. Runs inference with private inputs
3. Generates a zero-knowledge proof
4. Verifies the proof

**Proof time**: ~0.7s
**Proof size**: 524 bytes

## Privacy Properties

When using JOLT Atlas to prove authorization:

### Public Inputs
- Transaction amount (revealed)
- Vendor ID (revealed)
- Timestamp (revealed)

### Private Inputs
- Balance (hidden)
- Velocity data (hidden)
- Vendor trust score (hidden)

### Proof Output
- "Authorized: YES/NO" (revealed)
- Risk score (revealed)
- **Balance, velocity, trust scores remain private**

## Scaling Factors

To maintain numerical stability in the neural network:

- **Amount**: Divide by 1,000,000 (convert micro-USDC to millions)
- **Balance**: Divide by 1,000,000
- **Velocity**: Divide by 1,000,000
- **Trust**: Already 0-1, no scaling needed

Example:
- Real amount: 50,000 micro-USDC
- Scaled input: 0.05

## Limitations (ONNX / JOLT Atlas)

✅ **Supported**:
- Feedforward neural networks
- Linear layers, ReLU, Sigmoid, Tanh
- Numeric comparisons (via learned weights)
- Batch normalization, pooling

❌ **Not Supported**:
- If/else branching (use zkEngine WASM instead)
- Loops or recursion
- String operations (vendor whitelist checking)
- External API calls

For complex policies with these requirements, use the zkEngine WASM prover instead (see `../../zkengine-prover/`).

## Next Steps

1. **Experiment with architecture**:
   - Try deeper networks (more layers)
   - Adjust hidden layer sizes
   - Add dropout for regularization

2. **Customize policy rules**:
   - Adjust velocity thresholds
   - Add more features (time-of-day, vendor category)
   - Multi-class classification (approve/review/reject)

3. **Generate proofs**:
   ```bash
   cd ../../jolt-prover
   cargo run --release --example simple_auth
   ```

4. **Integrate with x402**:
   - Use proofs in `X-Agent-Auth-Proof` header
   - Stack with Fair-Pricing proofs
   - Deploy to production

## References

- [JOLT Atlas (zkML)](https://github.com/ICME-Lab/zkml-jolt)
- [ONNX Specification](https://onnx.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
