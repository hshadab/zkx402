# zkX402: Verifiable Agent Authorization for x402

Trustless, verifiable authorization for AI agents in the x402 payment protocol using JOLT Atlas ONNX inference proofs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![JOLT Atlas](https://img.shields.io/badge/JOLT-Atlas-blue.svg)](https://github.com/ICME-Lab/jolt-atlas)
[![x402](https://img.shields.io/badge/x402-payment%20protocol-green.svg)](https://x402.org)

## 🚀 What is zkX402?

zkX402 brings **verifiable authorization to the x402 payment protocol**, enabling AI agents to generate cryptographic proofs that their authorization decisions were computed correctly. Using JOLT Atlas ONNX inference proofs, agents can demonstrate that policies were executed faithfully - making authorization **trustless**, **tamper-proof**, and **auditable**.

**The Innovation**: Zero-knowledge proofs of correct ONNX model inference provide cryptographic guarantees that authorization logic executed correctly. The x402 ecosystem doesn't need to trust the agent's claim of "approved" - they can **verify** it cryptographically.

**What zkX402 Adds to x402**:
- ✅ **Verifiable Inference**: Cryptographic proof that the ML model executed correctly with given inputs
- ✅ **Trustless Authorization**: Don't trust agent decisions - verify them cryptographically
- ✅ **Tamper-Proof Policies**: Can't fake or manipulate authorization logic execution
- ✅ **Auditable Decisions**: Anyone can verify proofs match claimed authorization outcomes
- ✅ **Transparent Enforcement**: Policies are provably executed as specified

**x402 Integration Use Cases**:
- 💰 **Verifiable Agent Spending**: Cryptographically prove spending policy was checked correctly
- 🔐 **Trustless Payment Gates**: Verify authorization logic executed faithfully in x402 flows
- 📊 **Auditable Risk Assessment**: Proof that fraud detection model ran correctly
- 🤝 **Compliance Verification**: Provably demonstrate policy adherence for x402 transactions
- 🏦 **Trustless Agent Commerce**: Enable verifiable agent-to-agent x402 payments without trusted intermediaries

## ✨ Features

- ✅ **ONNX Inference Proofs**: Zero-knowledge proofs of correct ML model execution (~0.7s generation)
- ✅ **x402 Protocol Integration**: Full HTTP 402 payment flow with Base USDC payments
- ✅ **Base L2 Payments**: Real stablecoin payments on Base mainnet with on-chain verification
- ✅ **Rule-Based Policies**: Threshold checks, comparisons, velocity limits for payment authorization
- ✅ **Neural Network Policies**: ML-based risk scoring and classification for x402 transactions
- ✅ **Web UI**: Interactive proof generation with free tier (5 proofs/day) and model comparison
- ✅ **REST API**: Production-ready external API with rate limiting and payment verification
- ✅ **Model Registry**: Upload and manage custom ONNX authorization models
- ✅ **Proof History**: Persistent history with export functionality
- ✅ **Batch Processing**: Generate multiple proofs in parallel for high-throughput x402 systems
- ✅ **Comprehensive Tests**: Full E2E test coverage (Jest + Rust)

## 💳 Payment System

zkX402 accepts **real USDC stablecoin payments on Base L2** for production x402 agent authorization proofs.

### Payment Details

- **Network**: Base Mainnet (Chain ID: 8453)
- **Token**: USDC at `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`
- **Payment Wallet**: `0x1f409E94684804e5158561090Ced8941B47B0CC6`
- **Explorer**: [View on BaseScan](https://basescan.org/address/0x1f409E94684804e5158561090Ced8941B47B0CC6)

### Pricing

All 14 curated models have transparent, pay-per-proof pricing:

| Model Tier | Price Range | Use Cases |
|------------|-------------|-----------|
| **Test Models** | $0.005 | Operation verification, testing |
| **Basic Auth** | $0.01 - $0.015 | Simple threshold checks, vendor trust |
| **Velocity Control** | $0.02 | Hourly/daily spending limits |
| **Advanced** | $0.04 - $0.06 | Multi-factor auth, neural risk scoring |

**See complete pricing**: [PAYMENT_GUIDE.md](zkx402-agent-auth/PAYMENT_GUIDE.md)

### Free Tier vs Paid Tier

| Tier | Access | Limit | Use Case |
|------|--------|-------|----------|
| **Free** | Web UI | 5 proofs/day | Testing, demonstrations |
| **Paid** | x402 API | Unlimited | Production AI agents |

### Payment Flow for x402 Agents

```bash
# 1. Discover payment requirements
curl http://localhost:3001/.well-known/x402

# 2. Request authorization (receive 402 Payment Required)
curl -X POST http://localhost:3001/x402/authorize/simple_threshold

# 3. Send USDC payment on Base (using ethers.js or web3.js)
# Transaction to: 0x1f409E94684804e5158561090Ced8941B47B0CC6

# 4. Generate zkML proof locally

# 5. Submit payment + proof via X-PAYMENT header
curl -X POST http://localhost:3001/x402/authorize/simple_threshold \
  -H "X-PAYMENT: <base64-encoded-payment-and-proof>"

# 6. Receive authorization result with X-PAYMENT-RESPONSE header
```

**Complete integration guide**: [PAYMENT_GUIDE.md](zkx402-agent-auth/PAYMENT_GUIDE.md)

## 🎯 Quick Start

### Prerequisites

- **Rust**: 1.70+ ([Install](https://rustup.rs/))
- **Node.js**: 20+ ([Install](https://nodejs.org/))
- **Python**: 3.8+ (for ONNX model generation)

### Installation

```bash
# Clone repository
git clone https://github.com/hshadab/zkx402.git
cd zkx402/zkx402-agent-auth

# Generate ONNX models
cd policy-examples/onnx
python3 create_demo_models.py
cd ../..

# Test proof generation (Rust)
cd jolt-prover
cargo run --release --example integer_auth_e2e

# Start web UI
cd ../ui
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

### 5-Minute Demo

**1. Generate a Proof (Web UI)**

Navigate to http://localhost:3000, select "Simple Auth" model, and click "Generate Proof" with default inputs:
- Amount: 50 ($0.50)
- Balance: 1000 ($10.00)
- Vendor Trust: 80 (0.80)

Result: **✅ APPROVED** in ~0.7 seconds

**2. Generate a Proof (API)**

```bash
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_auth",
    "inputs": {
      "amount": "50",
      "balance": "1000",
      "velocity_1h": "20",
      "velocity_24h": "100",
      "vendor_trust": "80"
    }
  }'
```

**3. Generate a Proof (Rust)**

```rust
use zkx402_jolt_auth::*;

let inputs = AuthInputs {
    amount: 50,
    balance: 1000,
    velocity_1h: 20,
    velocity_24h: 100,
    vendor_trust: 80,
};

let proof = generate_proof("simple_auth.onnx", inputs)?;
println!("Approved: {}", proof.approved);
```

## 🎯 Production-Ready Curated Models

We provide **14 curated ONNX authorization models** (10 production + 4 test) covering the most common x402 payment authorization use cases. These models are fully tested, validated, and ready to use with JOLT Atlas proofs.

### What Are These Models?

Think of these as **pre-built authorization rules that agents can prove they've followed**—without revealing private payment data. For example:

- An AI agent can prove "I checked that the account balance was sufficient" without revealing the actual balance
- An agent can prove "The vendor trust score met the minimum threshold" without revealing the vendor's identity
- An agent can prove "Spending velocity was within limits" without revealing transaction history

### How They Work (Plain English)

1. **The Setup**: You choose an authorization model (e.g., "simple_threshold" for basic balance checks)
2. **The Check**: The agent runs the model with private x402 payment inputs (amount=$50, balance=$100)
3. **The Proof**: JOLT Atlas generates a zero-knowledge proof showing the model approved the transaction
4. **The Verification**: Anyone in the x402 ecosystem can verify the proof without seeing the private inputs
5. **The Result**: Trust without transparency—privacy preserved, authorization verified

### Available Models

#### Production Models (10)

| Model | What It Does | When To Use |
|-------|-------------|-------------|
| **simple_threshold** | Checks if you have enough money | Basic wallet balance checks in x402 payments |
| **percentage_limit** | Limits spending to X% of balance | "Don't spend more than 10% at once" policies |
| **vendor_trust** | Requires minimum vendor reputation | x402 marketplace transactions |
| **velocity_1h** | Limits spending per hour | Rate limiting, fraud prevention for x402 agents |
| **velocity_24h** | Limits spending per day | Daily spending caps for x402 payments |
| **daily_limit** | Hard cap on daily spending | Budget enforcement for agent spending |
| **age_gate** | Checks minimum age | Age-restricted x402 purchases |
| **multi_factor** | Combines balance + velocity + trust | High-security x402 transactions |
| **composite_scoring** | Weighted risk score from multiple factors | Advanced risk assessment for x402 |
| **risk_neural** | ML-based risk scoring | Sophisticated fraud detection for agent payments |

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
cd zkx402-agent-auth/policy-examples/onnx/curated
python3 test_all_models.py

# 2. Use a specific model for x402 payment authorization
# Example: Check if $50 transaction is OK with $100 balance
cd ../../jolt-prover
./target/release/examples/proof_json_output \
  ../policy-examples/onnx/curated/simple_threshold.onnx \
  5000 10000  # $50 and $100 in cents
```

### Real-World x402 Example

**Scenario**: An AI shopping agent needs to buy office supplies ($75) via x402 but shouldn't reveal the company's bank balance.

**Without zkML**:
- ❌ Agent reveals balance to vendor: "I have $10,000, so $75 is fine"
- ❌ Privacy compromised, creates security risk in x402 payment flow

**With zkML (using simple_threshold.onnx)**:
- ✅ Agent generates proof: "I ran the authorization model and it approved"
- ✅ Vendor verifies proof in x402 flow: Valid ✓
- ✅ Balance stays private, authorization confirmed cryptographically

### Model Details

All 14 models are:
- ✅ **Validated**: Pass comprehensive test suites (29 test cases)
- ✅ **JOLT-Compatible**: Use only supported operations (Add, Sub, Mul, Div, Greater, Less, Cast, Clip, Identity, Slice)
- ✅ **Production-Ready**: Tested with ONNX Runtime and ready for x402 proof generation
- ✅ **Documented**: Full specifications available

**See detailed documentation**:
- [README.md](zkx402-agent-auth/policy-examples/onnx/curated/README.md) - Quick start guide for all 14 curated models
- [CATALOG.md](zkx402-agent-auth/policy-examples/onnx/curated/CATALOG.md) - Complete specifications for all 14 models (10 production + 4 test)
- [TEST_RESULTS.md](zkx402-agent-auth/policy-examples/onnx/curated/TEST_RESULTS.md) - Test results and benchmarks
- [JOLT_ENHANCEMENT_USAGE.md](zkx402-agent-auth/policy-examples/onnx/curated/JOLT_ENHANCEMENT_USAGE.md) - Which enhancements each model uses

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Get started in 5 minutes
- **[API_REFERENCE.md](API_REFERENCE.md)**: Complete API documentation
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Production deployment guide
- **[JOLT_ATLAS_ENHANCEMENTS.md](zkx402-agent-auth/jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md)**: Technical details on JOLT Atlas enhancements

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      zkX402 System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Web UI     │  │  REST API    │  │   Python     │    │
│  │  (React)     │  │  (Express)   │  │   Client     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            ▼                                │
│                  ┌──────────────────┐                       │
│                  │  Node.js API     │                       │
│                  │  - Model Registry│                       │
│                  │  - Proof History │                       │
│                  │  - Validation    │                       │
│                  └─────────┬────────┘                       │
│                            │                                │
│                            ▼                                │
│                  ┌──────────────────┐                       │
│                  │  Rust Prover     │                       │
│                  │  (JOLT Atlas)    │                       │
│                  │  - ONNX Tracer   │                       │
│                  │  - ZK Proof Gen  │                       │
│                  │  - Verification  │                       │
│                  └──────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🎨 Components

### 1. Enhanced JOLT Atlas ONNX Inference Prover

We've extended JOLT Atlas to support practical authorization use cases for x402 payment authorization:

**Enhancements Made**:
- **Comparison Operations**: `Greater`, `Less`, `GreaterEqual` for threshold-based payment authorization
- **Arithmetic Operations**: `Div` (division) and `Cast` (type conversion) for advanced scoring models
- **Tensor Operations**: `Slice`, `Identity`, `Reshape` for complex policy data manipulation
- **MatMult Enhancements**: Extended support for 1D tensor outputs in neural network policies
- **Increased Tensor Size Limits**: `MAX_TENSOR_SIZE` increased from 64→1024 to support larger authorization models (e.g., 18 features × 32-bit weights = 576 elements)

**What This Enables for x402**:
- ✅ **Verifiable Payment Authorization**: Cryptographically prove spending limit checks executed correctly
- ✅ **Trustless Rule-Based Policies**: Verifiable threshold checks for x402 transactions (amount < 10% of balance)
- ✅ **Auditable Neural Network Decisions**: Provably demonstrate ML risk scoring was computed correctly
- ✅ **Tamper-Proof Velocity Controls**: Cryptographic proof that rate limiting was enforced faithfully
- ✅ **Transparent Multi-Criteria Logic**: Verifiable complex authorization with zero-knowledge proofs

**Core Innovation**: These enhancements enable **verifiable ONNX inference** - cryptographic proofs that an ONNX model was executed correctly with specific inputs, producing a specific output. The x402 ecosystem can verify authorization decisions cryptographically rather than trusting agent claims, making authorization **trustless** and **auditable**.

See [JOLT_ATLAS_ENHANCEMENTS.md](zkx402-agent-auth/jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md) for technical details.

### 2. Web UI

Interactive interface for testing x402 authorization policies:

- **Authorization Simulator**: Test x402 payment policies with custom inputs
- **Model Comparison**: Compare authorization results across different policy models
- **Proof History**: View and export previous authorization proofs
- **Model Registry**: Upload and manage custom ONNX authorization models for x402
- **Real-Time Progress**: Animated loading states with inference proof generation stages

**Start UI**:
```bash
cd zkx402-agent-auth/ui
npm install
npm run dev
```

### 3. REST API Server for x402 Integration

Production-ready API for integrating zkX402 authorization into x402 payment systems:

**Endpoints**:
```
GET  /api/v1/health            - Health check
GET  /api/v1/models            - List authorization models
POST /api/v1/proof             - Generate single inference proof
POST /api/v1/proof/batch       - Batch proof generation for high-throughput x402 systems
```

**Features**:
- Rate limiting (100 req/15min) for production x402 services
- Request ID tracking for audit trails
- Structured logging for x402 transaction monitoring
- Input validation for payment authorization requests
- CORS support for web-based x402 applications

**Start API**:
```bash
cd zkx402-agent-auth/api-server
npm install
npm start
```

### 4. Model Registry

Upload and manage custom x402 authorization models:

```bash
# Via UI: Drag & drop .onnx files for custom x402 payment policies

# Via API:
curl -X POST http://localhost:3001/api/upload-model \
  -F "model=@my_x402_policy.onnx" \
  -F "description=Custom x402 payment authorization policy"
```

## 📊 Performance

ONNX inference proof generation performance for x402 authorization:

| Model Type | Inference Proof Generation | Verification | Proof Size | Operations |
|------------|---------------------------|--------------|------------|-----------|
| Simple Auth | 0.7s | 45ms | 15 KB | 21 |
| Neural Network | 1.5s | 65ms | 40 KB | 45 |
| Complex NN | 3.0s | 100ms | 80 KB | 95 |

*Measured on: Intel i7, 16GB RAM*

**Optimizations for x402 Integration**:
- Rust release builds with LTO for production x402 services
- Dory polynomial commitment scheme for efficient proof generation
- Integer-only arithmetic (no floating point) for deterministic authorization
- Efficient ONNX operation tracing for fast inference proofs
- Suitable for real-time x402 payment authorization (<1s for most policies)

## 🔐 x402 Authorization Policies

### Rule-Based Policy for x402 Payments

```python
import torch

class X402RuleBasedAuth(torch.nn.Module):
    """
    Rule-based authorization for x402 payments.
    Generates verifiable inference proof that authorization logic executed correctly.
    """
    def forward(self, amount, balance, velocity_1h, velocity_24h, trust):
        # Scale to integers (100 = 1.00) for deterministic ZK proofs
        amount_i = (amount * 100).int()
        balance_i = (balance * 100).int()

        # x402 payment authorization rules
        rule1 = amount_i < (balance_i * 10 // 100)  # amount < 10% balance
        rule2 = (trust * 100).int() > 50             # vendor trust > 0.5
        rule3 = (velocity_1h * 100).int() < (balance_i * 5 // 100)  # hourly limit

        # All must pass for x402 payment authorization
        approved = (rule1 & rule2 & rule3).int()
        return approved

# Export to ONNX for inference proofs
torch.onnx.export(model, inputs, "x402_rule_based.onnx", opset_version=14)
```

### Neural Network Policy for x402 Payments

```python
class X402NeuralAuth(torch.nn.Module):
    """
    Neural network authorization for x402 payments.
    ML-based risk scoring with inference proof generation.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8)
        self.fc2 = torch.nn.Linear(8, 4)
        self.fc3 = torch.nn.Linear(4, 1)

    def forward(self, amount, balance, velocity_1h, velocity_24h, trust):
        # Stack inputs (already scaled to integers for ZK proofs)
        x = torch.stack([amount, balance, velocity_1h, velocity_24h, trust])

        # Neural network inference (proven via JOLT Atlas)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        score = self.fc3(x)

        # Threshold for x402 payment authorization
        approved = (score > 0.5).int()
        return approved
```

**Key Point**: These ONNX models generate **verifiable inference proofs** - cryptographic proofs that the model was executed correctly with specific inputs (balance, velocity), producing a verifiable authorization decision. The x402 ecosystem can cryptographically verify the decision is correct rather than trusting the agent's claim.

## 🧪 Testing

### Run All Tests

```bash
# API tests (Jest)
cd zkx402-agent-auth/ui
npm test

# Rust integration tests
cd zkx402-agent-auth/jolt-prover
cargo test

# End-to-end test
cd zkx402-agent-auth/jolt-prover
cargo run --release --example integer_auth_e2e
```

### Test Coverage

- ✅ Proof generation (approved/rejected cases)
- ✅ Model validation
- ✅ API endpoints
- ✅ Batch processing
- ✅ Error handling
- ✅ Performance benchmarks

## 📦 Project Structure

```
zkx402/
├── README.md                           # This file
├── QUICKSTART.md                       # 5-minute guide
├── API_REFERENCE.md                    # API documentation
├── DEPLOYMENT.md                       # Deployment guide
├── IMPROVEMENTS_SUMMARY.md             # Recent improvements
└── zkx402-agent-auth/
    ├── jolt-atlas-fork/                # Enhanced JOLT Atlas
    │   ├── JOLT_ATLAS_ENHANCEMENTS.md # Technical docs
    │   ├── onnx-tracer/               # ONNX tracer with new ops
    │   └── zkml-jolt-core/            # Core zkVM
    ├── jolt-prover/                    # Proof generation
    │   ├── examples/
    │   │   ├── integer_auth_e2e.rs    # Main example
    │   │   ├── proof_json_output.rs   # JSON output for API
    │   │   └── velocity_auth.rs       # Velocity checks
    │   └── tests/                     # Integration tests
    ├── policy-examples/onnx/           # Model generation
    │   ├── create_demo_models.py      # Generate demos
    │   ├── test_models.py             # Validate models
    │   └── *.onnx                     # Pre-built models
    ├── ui/                             # React web interface
    │   ├── src/components/
    │   │   ├── AuthorizationSimulator.jsx
    │   │   ├── ProofHistory.jsx       # Proof history
    │   │   ├── ModelComparison.jsx    # Model comparison
    │   │   ├── ModelRegistry.jsx      # Model upload
    │   │   └── LoadingIndicator.jsx   # Progress animation
    │   ├── tests/                     # Jest tests
    │   └── server.js                  # Express API
    └── api-server/                     # External REST API
        ├── server.js                   # Production API
        └── README.md                   # API docs
```

## 🌐 x402 Integration Examples

### JavaScript/TypeScript - x402 Payment Flow

```javascript
const axios = require('axios');

/**
 * Generate verifiable inference proof for x402 payment authorization
 * @param {number} amount - Payment amount in x402 transaction
 * @param {number} balance - Agent's balance used in authorization check
 * @param {number} trust - Vendor trust score
 * @returns {Object} Verifiable proof of authorization for x402 payment
 */
async function authorizeX402Payment(amount, balance, trust) {
  const response = await axios.post('http://localhost:4000/api/v1/proof', {
    model: 'simple_auth',
    inputs: {
      amount: amount.toString(),
      balance: balance.toString(),
      velocity_1h: '20',
      velocity_24h: '100',
      vendor_trust: trust.toString()
    }
  });

  // Returns verifiable proof that authorization logic executed correctly
  // x402 recipients can cryptographically verify the decision is valid
  return {
    approved: response.data.approved,
    zkProof: response.data.zkmlProof,  // Attach to x402 payment for verification
    requestId: response.data.requestId
  };
}

// Usage in x402 payment flow
const authProof = await authorizeX402Payment(50, 1000, 80);
console.log('x402 Payment Authorized:', authProof.approved);
console.log('Attach proof to x402 transaction:', authProof.zkProof);
```

### Python - x402 Service Integration

```python
import requests

def authorize_x402_payment(amount, balance, trust):
    """
    Generate verifiable inference proof for x402 payment authorization.

    Args:
        amount: Payment amount in x402 transaction
        balance: Agent's balance used in authorization check
        trust: Vendor trust score

    Returns:
        dict: Verifiable proof of authorization for x402 payment
    """
    response = requests.post('http://localhost:4000/api/v1/proof', json={
        'model': 'simple_auth',
        'inputs': {
            'amount': str(amount),
            'balance': str(balance),
            'velocity_1h': '20',
            'velocity_24h': '100',
            'vendor_trust': str(trust)
        }
    })

    result = response.json()
    return {
        'approved': result['approved'],
        'inference_proof': result['zkmlProof'],  # Attach to x402 payment
        'proof_size': result['proofSize'],
        'verification_time': result['verificationTime']
    }

# Usage in x402 payment system
proof = authorize_x402_payment(50, 1000, 80)
print(f'x402 Payment Authorized: {proof["approved"]}')
print(f'Proof Size: {proof["proof_size"]}')
print(f'Inference Proof: {proof["inference_proof"]["commitment"][:32]}...')
```

### Rust - Native x402 Integration

```rust
use zkml_jolt_core::jolt::*;
use onnx_tracer::{model, tensor::Tensor};

/// Generate verifiable ONNX inference proof for x402 payment authorization
///
/// # Arguments
/// * `amount` - Payment amount in x402 transaction
/// * `balance` - Agent's balance used in authorization check
/// * `trust` - Vendor trust score
///
/// # Returns
/// * Verifiable proof that authorization logic executed correctly
fn authorize_x402_payment(amount: i32, balance: i32, trust: i32) -> Result<(bool, JoltProof)> {
    let model_obj = model(&"simple_auth.onnx".into());
    let inputs = vec![amount, balance, 20, 100, trust];
    let input_tensor = Tensor::new(Some(&inputs), &vec![1, 5])?;

    // Generate inference proof via JOLT Atlas
    let (proof, output) = generate_and_verify_proof(model_obj, input_tensor)?;

    // Authorization decision from model inference
    let approved = output.inner[0] > 50;

    Ok((approved, proof))
}

// Usage in x402 payment flow
let (approved, inference_proof) = authorize_x402_payment(50, 1000, 80)?;
println!("x402 Payment Authorized: {}", approved);
println!("Inference Proof Generated: {} bytes", inference_proof.size());
// Attach inference_proof to x402 payment transaction
```

## 🚀 Deployment

### Docker

```bash
docker-compose up -d
```

### Railway

```bash
railway init
railway up
```

### Manual (Ubuntu)

```bash
# Install dependencies
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs python3-pip

# Build and run
cd zkx402-agent-auth
./deploy.sh
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete instructions.

## 📈 Roadmap

### ✅ Completed
- [x] Enhanced JOLT Atlas with comparison operations and increased tensor sizes
- [x] ONNX inference proof generation
- [x] Rule-based authorization policies for x402
- [x] Neural network authorization policies for x402
- [x] Web UI with proof history
- [x] REST API for x402 service integration
- [x] Model registry and upload
- [x] Comprehensive test suite
- [x] Production deployment guides

### 🚧 In Progress
- [ ] x402 protocol integration examples
- [ ] x402 Bazaar marketplace integration
- [ ] Multi-agent x402 commerce demonstrations

### 🔮 Future
- [ ] N-API bindings for faster inference proof generation
- [ ] Proof caching by input hash for high-throughput x402 systems
- [ ] WebSocket support for real-time x402 authorization updates
- [ ] Standalone verifier application for x402 proof verification
- [ ] Model training pipeline in UI for custom x402 policies
- [ ] Multi-tenant support for x402 service providers

## 🤝 Contributing

Contributions welcome! To add new ONNX operations:

1. Fork the repository
2. Add operation to `jolt-atlas-fork/onnx-tracer/src/trace_types.rs`
3. Implement zkVM instruction if needed
4. Add test cases
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

This project builds on [JOLT Atlas](https://github.com/ICME-Lab/jolt-atlas) (MIT License).

## 🙏 Acknowledgments

- **x402 Protocol**: Payment protocol for AI agents
- **JOLT Atlas Team**: ONNX inference proof framework
- **ONNX Community**: Model format and tooling
- **Rust Crypto Community**: Cryptographic primitives

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hshadab/zkx402/issues)
- **Documentation**: [Full Docs](https://github.com/hshadab/zkx402/tree/main/docs)
- **x402 Protocol**: [x402.org](https://x402.org)
- **JOLT Atlas**: [Original Project](https://github.com/ICME-Lab/jolt-atlas)

---

**Built with** ❤️ **for the x402 ecosystem using JOLT Atlas ONNX inference proofs**

**Status**: 🚀 Production Ready for x402 Integration
