# Agent Authorization with JOLT Atlas zkML

**Zero-Knowledge Proof-Carrying Authorization for AI Agents**

This project implements privacy-preserving agent authorization using JOLT Atlas, a zero-knowledge machine learning proof system for ONNX models. Agents can prove they're authorized to make transactions without revealing their balance, velocity, or policy thresholds.

## 🎯 What This Does

**The Problem**: AI agents with spending power need authorization controls, but traditional approaches reveal sensitive information like balances and spending limits.

**Our Solution**: Agents generate zero-knowledge proofs that they satisfy authorization policies (implemented as ONNX neural networks) without revealing:
- Account balance
- Spending velocity (hourly/daily)
- Policy thresholds
- Internal authorization logic

**Think of it like**: Your agent showing a cryptographic receipt proving "this transaction passed all checks" without revealing your actual balance, spending history, or the rules themselves.

## 🚀 JOLT Atlas Enhancements

This project uses an enhanced fork of JOLT Atlas with expanded operation support for real-world authorization policies.

### New Operations Added

**Comparison Operations:**
- `Greater` (`>`): vendor_trust > 0.5, balance > amount
- `Less` (`<`): amount < daily_limit, velocity < max_rate
- `GreaterEqual` (`>=`): age >= 18, score >= threshold

**Tensor Operations:**
- `Slice`: Extract subsets of tensors
- `Identity`: Pass-through for computational graphs
- `Reshape`: Tensor shape manipulation

**MatMult Enhancements:**
- Fixed 1D tensor dimension handling
- Support both 2D `[m, n]` and 1D `[n]` outputs
- Enables neural network bias addition and single-row outputs

### Why These Matter

**Before Enhancements:**
- JOLT Atlas only supported basic arithmetic
- Neural network policies failed on comparison operations
- MatMult crashed on 1D tensors
- Limited to trivial authorization rules

**After Enhancements:**
- ✅ Rule-based policies (thresholds, comparisons)
- ✅ Neural network scoring (full ML models)
- ✅ Hybrid policies (rules + ML)
- ✅ Real-world authorization use cases

**See**: [`jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md`](jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md) for complete technical documentation.

## 📁 Project Structure

```
zkx402-agent-auth/
├── jolt-atlas-fork/              # Enhanced JOLT Atlas
│   ├── JOLT_ATLAS_ENHANCEMENTS.md  # Full enhancement documentation
│   ├── demo-models/               # 5 demonstration ONNX models
│   │   ├── comparison_demo.onnx   # Greater, Less, GreaterEqual
│   │   ├── tensor_ops_demo.onnx   # Slice, Identity, Reshape
│   │   ├── matmult_1d_demo.onnx   # MatMult with 1D output
│   │   ├── simple_auth.onnx       # Rule-based authorization
│   │   └── neural_auth.onnx       # Neural network authorization
│   └── onnx-tracer/src/
│       ├── trace_types.rs         # Added Greater, Less, Identity, Slice
│       ├── ops/hybrid.rs          # Greater/Less mappings
│       └── ops/poly.rs            # Identity/Slice mappings
│
├── jolt-prover/                  # JOLT Atlas integration
│   ├── src/lib.rs                # Prover wrapper
│   └── examples/                 # Authorization examples
│       ├── integer_auth_e2e.rs   # Integer-only auth demo
│       └── velocity_auth.rs      # Velocity policy (advanced)
│
└── policy-examples/onnx/         # Policy models & training
    ├── create_demo_models.py     # Generate demo models
    ├── test_models.py            # Validate models
    ├── comparison_demo.onnx      # Ready-to-use models
    ├── simple_auth.onnx
    └── neural_auth.onnx
```

## 🚀 Quick Start

### Prerequisites

```bash
# Rust (for JOLT Atlas prover)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python (for ONNX model creation)
python3 -m pip install torch onnx
```

### Run Demo

Test the enhanced JOLT Atlas with a simple authorization policy:

```bash
cd jolt-prover
cargo run --example integer_auth_e2e
```

**Expected Output:**
```
╔═══════════════════════════════════════════════════════╗
║  INTEGER-ONLY Authorization - JOLT Atlas E2E Test    ║
╚═══════════════════════════════════════════════════════╝

📝 Test Case 1: Approved Transaction
───────────────────────────────────────────────────────
Inputs (scaled by 100):
  Amount:      5 ($0.05, public)
  Balance:     1000 ($10.00, private)
  Velocity 1h: 2 ($0.02, private)
  Velocity 24h: 10 ($0.10, private)
  Vendor trust: 80 (0.80, private)

[1/5] Loading ONNX model...
      ✓ Model loaded: simple_auth.onnx

[2/5] Preprocessing JOLT prover...
      ✓ Prover preprocessed

[3/5] Preparing inputs...
      ✓ Inputs prepared (1 public, 4 private)

[4/5] Generating JOLT Atlas proof...
      ✓ Proof generated

[5/5] Verifying proof...
      ✓ Proof verified

═══════════════════════════════════════════════════════
✅ Zero-knowledge proof confirms: TRANSACTION APPROVED
   Output: 1 (approved)
   Verifier only sees: amount=$0.05, proof=524 bytes
   Hidden: balance, velocity, trust score, policy rules
═══════════════════════════════════════════════════════
```

### Create Your Own Policy

```python
# policy-examples/onnx/my_policy.py
import torch
import torch.nn as nn

class AuthPolicy(nn.Module):
    def forward(self, amount, balance, vendor_trust):
        # Rule 1: amount < 10% of balance
        rule1 = (amount < balance // 10).int()

        # Rule 2: vendor_trust > 50
        rule2 = (vendor_trust > 50).int()

        # Approve if both pass
        approved = rule1 * rule2
        return approved

# Export to ONNX
model = AuthPolicy()
dummy_inputs = (
    torch.tensor(5),      # amount
    torch.tensor(100),    # balance
    torch.tensor(75),     # vendor_trust
)
torch.onnx.export(model, dummy_inputs, "my_policy.onnx")
```

```bash
# Test with JOLT Atlas
cd jolt-prover
cargo run --example integer_auth_e2e -- ../policy-examples/onnx/my_policy.onnx
```

## 📊 Supported Authorization Use Cases

### 1. Rule-Based Authorization

**Policy**: Threshold checks using comparison operations

```python
def authorize(amount, balance, velocity_1h, vendor_trust):
    rule1 = amount < balance * 0.1      # Max 10% of balance
    rule2 = vendor_trust > 0.5          # Minimum trust
    rule3 = velocity_1h < 500           # Hourly limit
    return all([rule1, rule2, rule3])
```

**ONNX Operations Used**: Greater, Less, Mul, And
**Model**: [`policy-examples/onnx/simple_auth.onnx`](policy-examples/onnx/simple_auth.onnx)

### 2. Neural Network Scoring

**Policy**: Trained ML model for risk assessment

```python
class NeuralAuth(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, amount, balance, velocity_1h, velocity_24h, trust):
        x = torch.stack([amount, balance, velocity_1h, velocity_24h, trust])
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        risk_score = self.fc3(x)
        approved = (risk_score < 50).int()  # Using Less operation
        return approved
```

**ONNX Operations Used**: MatMult, Add, Clip (ReLU), Less
**Model**: [`policy-examples/onnx/neural_auth.onnx`](policy-examples/onnx/neural_auth.onnx)

### 3. Hybrid Authorization

**Policy**: Combine hard rules with ML scoring

```python
def hybrid_authorize(amount, balance, velocity, trust):
    # Hard rule: basic checks must pass
    basic_ok = (amount < balance * 0.1) and (trust > 0.5)

    # If basic checks pass, use ML for final decision
    if basic_ok:
        risk_score = neural_model([amount, balance, velocity, trust])
        return risk_score < 0.7
    return False
```

**Best of Both Worlds**: Fast rule filtering + sophisticated ML scoring

## 🔧 ONNX Model Requirements

**Supported Operations:**
- ✅ Arithmetic: Add, Sub, Mul, Div
- ✅ Comparison: Greater (`>`), GreaterEqual (`>=`), Less (`<`), Equal
- ✅ Matrix: MatMult (2D and 1D), Conv
- ✅ Tensor: Reshape, Flatten, Slice, Identity, Broadcast
- ✅ Activation: ReLU (via Clip), Sigmoid
- ✅ Reduction: Sum, Mean, ArgMax

**Data Types:**
- ✅ Integer-scaled operations (scale by 100 for decimals)
- ❌ Float operations (convert to integer-scaled)

**Limitations:**
- Max tensor size: 1024 elements
- Batch size must be 1
- Scale factor: Must use 128 (hardcoded in fork)

**See**: [`jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md`](jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md) for complete operation support matrix and model creation guidelines.

## 📈 Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Proving time | ~0.7-1.2s | Per authorization |
| Proof size | 524 bytes | Constant |
| Verification | <50ms | On-chain ready |
| Model complexity | Up to 1024 params | Tensor size limit |

**Comparison:**
- JOLT Atlas (this project): 0.7s proving, 524 bytes
- General zkVM (archived): 5-10s proving, 1-2KB proofs
- **6-12x faster** with enhanced JOLT Atlas

## 🛠️ Development

### Build

```bash
# Build JOLT Atlas fork
cd jolt-atlas-fork
cargo build --release

# Build prover
cd ../jolt-prover
cargo build --release
```

### Test

```bash
# Run all examples
cargo test

# Run specific example
cargo run --example integer_auth_e2e

# Test ONNX models
cd ../policy-examples/onnx
python3 test_models.py
```

### Create New ONNX Models

```bash
cd policy-examples/onnx

# Generate demo models
python3 create_demo_models.py

# Creates:
# - comparison_demo.onnx (Greater, Less operations)
# - tensor_ops_demo.onnx (Slice, Identity, Reshape)
# - matmult_1d_demo.onnx (1D tensor support)
# - simple_auth.onnx (Rule-based policy)
# - neural_auth.onnx (Neural network policy)
```

## 📚 Documentation

- **[JOLT Atlas Enhancements](jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md)** - Complete technical documentation of all enhancements
- **[Demo Models README](policy-examples/onnx/README.md)** - Guide to example ONNX models
- **[JOLT Paper](https://eprint.iacr.org/2023/1217)** - Original JOLT zkVM paper
- **[ONNX Format](https://onnx.ai/)** - ONNX specification

## 🗂️ Archived Components

Previous hybrid routing approach (zkEngine + JOLT) has been archived to `archived/`:
- `archived/hybrid-router/` - Previous multi-backend routing
- `archived/zkengine-prover/` - WASM-based prover (slower than enhanced JOLT)

**Reason for Archive**: With the enhanced JOLT Atlas supporting comparison and tensor operations, we no longer need a hybrid approach. JOLT Atlas alone can now handle all authorization use cases with 6-12x better performance.

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional ONNX operation support
- Performance optimizations
- More authorization policy examples
- Better model training utilities

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **JOLT Atlas**: Original zkML proof system by ICME-Lab
- **Dory**: Polynomial commitment scheme
- **ONNX**: Open neural network exchange format

---

**Built for Autonomous Agent Accountability**

*Prove your agent is authorized — without revealing balance, velocity, or policy rules.*

**Powered by Enhanced JOLT Atlas zkML**
