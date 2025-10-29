# zkX402: Agent Authorization with Zero-Knowledge Machine Learning

Privacy-preserving authorization for AI agents using enhanced JOLT Atlas zkML proofs.

## Overview

This project enables AI agents to prove they're authorized to perform actions (e.g., spend money, access resources) **without revealing private financial data** like account balances or transaction history. By using zero-knowledge proofs of ONNX-based authorization policies, agents can maintain privacy while demonstrating compliance.

**Key Innovation**: Extended JOLT Atlas to support real-world authorization policies through comparison operations, tensor manipulation, and enhanced matrix multiplication.

## JOLT Atlas Enhancements

We've enhanced the [JOLT Atlas](https://github.com/ICME-Lab/jolt-atlas) zero-knowledge proof system to support practical agent authorization use cases:

### New Operations Added

**Comparison Operations:**
- `Greater` (`>`): Check if values exceed thresholds (e.g., `balance > amount`, `trust_score > 0.5`)
- `Less` (`<`): Enforce limits (e.g., `amount < daily_limit`, `velocity < max_rate`)
- `GreaterEqual` (`>=`): Minimum requirements (e.g., `age >= 18`, `score >= threshold`)

**Tensor Operations:**
- `Slice`: Extract feature subsets from multi-dimensional data
- `Identity`: Pass-through for graph construction and residual connections
- `Reshape`: Tensor shape manipulation

**MatMult Enhancements:**
- Fixed crash on 1D tensor outputs (neural network bias addition, single-row outputs)
- Support both 2D `[m, n]` and 1D `[n]` tensor dimensions

### Why These Matter

**Before Enhancements:**
- ❌ JOLT Atlas limited to basic arithmetic operations
- ❌ Neural network policies failed on comparison operations
- ❌ MatMult crashed on 1D tensors
- ❌ Could only implement trivial authorization rules

**After Enhancements:**
- ✅ Rule-based policies with thresholds and comparisons
- ✅ Neural network scoring with full ML models
- ✅ Hybrid policies combining rules + ML
- ✅ Real-world authorization use cases

## Authorization Use Cases

### 1. Rule-Based Authorization

**Example Policy**: Approve transaction if:
- Amount < 10% of balance
- Vendor trust score > 0.5
- 1-hour spending velocity < limit

**ONNX Model**: `policy-examples/onnx/simple_auth.onnx` (21 operations, ~1.5 KB)

**Proof Performance**: ~0.7s generation, ~15 KB proof size

### 2. Neural Network Scoring

**Example Policy**: Trained ML model classifies transaction risk based on:
- Transaction amount
- Account balance
- Short-term velocity (1h)
- Long-term velocity (24h)
- Vendor trust score

**Architecture**: `[5 inputs] → [8 hidden] → [4 hidden] → [1 output]`

**ONNX Model**: `policy-examples/onnx/neural_auth.onnx` (30 operations, ~3 KB)

**Proof Performance**: ~1.5s generation, ~40 KB proof size

### 3. Hybrid Authorization

**Example Policy**: Combine hard rules with ML scoring:
1. Apply strict threshold checks (Greater/Less comparisons)
2. If basic rules pass, run neural network risk scorer
3. Final approval based on combined logic

**Benefit**: Balance interpretability (rule-based) with flexibility (ML)

## Production-Ready Curated Models

We provide **14 curated ONNX authorization models** (10 production + 4 test) covering the most common use cases. These models are fully tested, validated, and ready to use with JOLT Atlas proofs.

### What Are These Models?

Think of these as **pre-built authorization rules that agents can prove they've followed**—without revealing the private data involved. For example:

- An AI agent can prove "I checked that the account balance was sufficient" without revealing the actual balance amount
- An agent can prove "The vendor trust score met the minimum threshold" without revealing the vendor's identity
- An agent can prove "Spending velocity was within limits" without revealing transaction history

### How They Work (Plain English)

1. **The Setup**: You choose an authorization model (e.g., "simple_threshold" for basic balance checks)
2. **The Check**: The agent runs the model with private inputs (amount=$50, balance=$100)
3. **The Proof**: JOLT Atlas generates a zero-knowledge proof showing the model approved the transaction
4. **The Verification**: Anyone can verify the proof without seeing the private inputs
5. **The Result**: Trust without transparency—privacy preserved, authorization verified

### Available Models

#### Production Models (10)

| Model | What It Does | When To Use |
|-------|-------------|-------------|
| **simple_threshold** | Checks if you have enough money | Basic wallet balance checks |
| **percentage_limit** | Limits spending to X% of balance | "Don't spend more than 10% at once" |
| **vendor_trust** | Requires minimum vendor reputation | Marketplace transactions |
| **velocity_1h** | Limits spending per hour | Rate limiting, fraud prevention |
| **velocity_24h** | Limits spending per day | Daily spending caps |
| **daily_limit** | Hard cap on daily spending | Budget enforcement |
| **age_gate** | Checks minimum age | Age-restricted purchases |
| **multi_factor** | Combines balance + velocity + trust | High-security transactions |
| **composite_scoring** | Weighted risk score from multiple factors | Advanced risk assessment |
| **risk_neural** | ML-based risk scoring | Sophisticated fraud detection |

#### Test Models (4)

| Model | What It Does | When To Use |
|-------|-------------|-------------|
| **test_less** | Tests Less comparison operation | Operation testing, JOLT Atlas validation |
| **test_identity** | Tests Identity pass-through operation | Graph construction, residual connections |
| **test_clip** | Tests Clip operation (ReLU approximation) | Activation function testing |
| **test_slice** | Tests Slice tensor operation | Feature extraction, tensor manipulation |

### Quick Start with Curated Models

```bash
# 1. Test a model with sample inputs
cd policy-examples/onnx/curated
python3 test_all_models.py

# 2. Use a specific model
# Example: Check if $50 transaction is OK with $100 balance
cd ../../jolt-prover
./target/release/examples/proof_json_output \
  ../policy-examples/onnx/curated/simple_threshold.onnx \
  5000 10000  # $50 and $100 in cents
```

### Real-World Example

**Scenario**: An AI shopping agent needs to buy office supplies ($75) but shouldn't reveal the company's bank balance.

**Without zkML**:
- ❌ Agent reveals balance to vendor: "I have $10,000, so $75 is fine"
- ❌ Privacy compromised, creates security risk

**With zkML (using simple_threshold.onnx)**:
- ✅ Agent generates proof: "I ran the authorization model and it approved"
- ✅ Vendor verifies proof: Valid ✓
- ✅ Balance stays private, authorization confirmed

### Model Details

All 14 models are:
- ✅ **Validated**: Pass comprehensive test suites (29 test cases)
- ✅ **JOLT-Compatible**: Use only supported operations (Add, Sub, Mul, Div, Greater, Less, Cast, Clip, Identity, Slice)
- ✅ **Production-Ready**: Tested with ONNX Runtime and ready for proof generation
- ✅ **Documented**: Full specifications in [CATALOG.md](policy-examples/onnx/curated/CATALOG.md)

See detailed documentation:
- [README.md](policy-examples/onnx/curated/README.md) - Quick start guide
- [CATALOG.md](policy-examples/onnx/curated/CATALOG.md) - Complete model specifications
- [TEST_RESULTS.md](policy-examples/onnx/curated/TEST_RESULTS.md) - Test results and benchmarks
- [JOLT_ENHANCEMENT_USAGE.md](policy-examples/onnx/curated/JOLT_ENHANCEMENT_USAGE.md) - Which enhancements each model uses

## Quick Start

### 1. Generate Demo ONNX Models

```bash
cd policy-examples/onnx
python3 create_demo_models.py
```

This creates 5 demonstration models:
- `comparison_demo.onnx` - Greater, Less, GreaterEqual operations
- `tensor_ops_demo.onnx` - Slice, Identity, Reshape operations
- `matmult_1d_demo.onnx` - MatMult with 1D output tensors
- `simple_auth.onnx` - Rule-based authorization policy
- `neural_auth.onnx` - Neural network authorization policy

### 2. Validate Models

```bash
python3 test_models.py
```

Expected output: `✓ All models valid!`

### 3. Run Authorization Example

```bash
cd jolt-prover
cargo run --example integer_auth_e2e
```

This demonstrates:
- Loading an ONNX authorization policy
- Generating a zero-knowledge proof of compliance
- Verifying the proof without seeing private data

## Project Structure

```
zkx402-agent-auth/
├── README.md                          # This file
├── jolt-atlas-fork/                   # Enhanced JOLT Atlas zkML prover
│   ├── JOLT_ATLAS_ENHANCEMENTS.md    # Detailed technical documentation
│   ├── onnx-tracer/                  # ONNX model tracer (with new ops)
│   └── zkml-jolt-core/               # Core zkVM instructions
├── jolt-prover/                       # Authorization proof examples
│   ├── examples/
│   │   ├── integer_auth_e2e.rs       # Rule-based policy example
│   │   └── velocity_auth.rs          # Velocity check example
│   └── src/lib.rs                    # Proof generation library
├── policy-examples/onnx/              # ONNX model generation scripts
│   ├── curated/                      # ⭐ 14 curated models (10 production + 4 test)
│   │   ├── README.md                 # Quick start guide
│   │   ├── CATALOG.md                # Complete model specifications
│   │   ├── TEST_RESULTS.md           # Test results and benchmarks
│   │   ├── JOLT_ENHANCEMENT_USAGE.md # Enhancement usage analysis
│   │   ├── test_all_models.py        # Automated test suite
│   │   ├── generate_all_models.py    # Model regeneration script
│   │   └── *.onnx                    # 14 curated authorization models
│   ├── create_demo_models.py         # Generate all 5 demo models
│   ├── test_models.py                # Validate ONNX models
│   └── *.onnx                        # Pre-built demonstration models
└── archived/                          # Previous hybrid routing approach
    ├── hybrid-router/                # Multi-backend router (deprecated)
    └── zkengine-prover/              # WASM-based prover (deprecated)
```

## Documentation

### Main Documentation
- **[JOLT_ATLAS_ENHANCEMENTS.md](jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md)**: Complete technical documentation of all JOLT Atlas enhancements
  - Enhanced operations (Greater, Less, Div, Cast, Slice, Identity, MatMult 1D)
  - Tensor size limits and shape support
  - Model creation guidelines
  - Performance characteristics
  - Known limitations and workarounds

### Curated Models
- **[policy-examples/onnx/curated/README.md](policy-examples/onnx/curated/README.md)**: Quick start guide for all 14 curated models
- **[policy-examples/onnx/curated/CATALOG.md](policy-examples/onnx/curated/CATALOG.md)**: Complete specifications for all 14 models (10 production + 4 test)
- **[policy-examples/onnx/curated/TEST_RESULTS.md](policy-examples/onnx/curated/TEST_RESULTS.md)**: Validation results and performance benchmarks
- **[policy-examples/onnx/curated/JOLT_ENHANCEMENT_USAGE.md](policy-examples/onnx/curated/JOLT_ENHANCEMENT_USAGE.md)**: Analysis of which enhancements each model uses

### Examples
- **[policy-examples/onnx/README.md](policy-examples/onnx/README.md)**: Guide to creating ONNX authorization policies
- **[jolt-prover/README.md](jolt-prover/README.md)**: Integration guide for proof generation

## Performance

| Model Type | Operations | Proof Time | Proof Size | Verification |
|------------|-----------|-----------|-----------|--------------|
| Simple rules | 10-20 | ~0.5s | ~15 KB | ~0.1s |
| Medium neural net | 20-50 | ~1.5s | ~40 KB | ~0.3s |
| Complex neural net | 50-100 | ~3.0s | ~80 KB | ~0.6s |

*Measured on: Intel i7, 16GB RAM*

## Model Creation Guidelines

### Integer Scaling Required

JOLT Atlas uses fixed-point arithmetic. Scale float values by 100-128:

```python
import torch

class IntegerScaledPolicy(torch.nn.Module):
    def __init__(self, scale=100):
        super().__init__()
        self.scale = scale

    def forward(self, amount, balance):
        # Scale inputs: $0.05 → 5 (scale=100)
        amount_scaled = (amount * self.scale).int()
        balance_scaled = (balance * self.scale).int()

        # Integer comparison: 5 < 1000
        approved = (amount_scaled < balance_scaled).int()
        return approved
```

### Export to ONNX

```python
torch.onnx.export(
    model,
    dummy_inputs,
    "policy.onnx",
    opset_version=14,
    do_constant_folding=False
)
```

### Validate with JOLT Atlas

```rust
use zkx402_jolt_auth::*;

let (proof, output) = generate_proof("policy.onnx", inputs)?;
assert!(verify_proof(&proof, &output));
```

## Supported ONNX Operations

### Arithmetic
✅ Add, Sub, Mul (integer only)
✅ Div (full support with scale factor handling)
✅ Cast (type conversion with scale adjustment)

### Comparison
✅ Greater (`>`), GreaterEqual (`>=`), Less (`<`), Equal

### Matrix Operations
✅ MatMult (2D and 1D), Conv (limited)

### Tensor Manipulation
✅ Reshape, Flatten, Slice, Broadcast, Identity

### Activation Functions
✅ ReLU (via Clip), Sigmoid (approximated)

### Reduction
✅ Sum, Mean, ArgMax

See [JOLT_ATLAS_ENHANCEMENTS.md](jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md) for complete operation support and limitations.

## Archived Components

The `archived/` directory contains previous approaches that are no longer active:

- **hybrid-router**: Multi-backend routing system (zkEngine + JOLT Atlas)
- **zkengine-prover**: WASM-based prover (5-10s proving time)

These were replaced by enhanced JOLT Atlas, which provides 6-12x better performance (0.7s proving) while supporting all required authorization operations through native comparison and tensor ops.

## Contributing

To extend JOLT Atlas operation support:

1. Add opcode to `ONNXOpcode` enum in `jolt-atlas-fork/onnx-tracer/src/trace_types.rs`
2. Add bitflag mapping in `into_bitflag()` method
3. Add conversion in appropriate file (`utilities.rs`, `poly.rs`, or `hybrid.rs`)
4. Implement zkVM instruction if needed in `zkml-jolt-core/src/jolt/instruction/`
5. Test with example ONNX model

## License

This project builds on [JOLT Atlas](https://github.com/ICME-Lab/jolt-atlas) (MIT License).

## References

- Original JOLT Atlas: https://github.com/ICME-Lab/jolt-atlas
- ONNX Operations: https://onnx.ai/onnx/operators/
- Tract ONNX: https://github.com/sonos/tract
- X402 Agent Authorization: https://github.com/hshadab/zkx402
