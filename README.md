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

## 💡 What zkX402 Provides

zkX402 enables **privacy-preserving authorization with cryptographic guarantees** for x402 payment systems.

**Core Capabilities**:
- 🔒 **Zero-Knowledge Privacy**: Prove authorization without revealing balances, transaction history, or account details
- ✅ **Cryptographic Verification**: Generate tamper-proof proofs that anyone can verify
- 📊 **14 Production Models**: Pre-built policies for common authorization scenarios
- 💳 **Real Blockchain Payments**: USDC on Base L2 with on-chain verification
- 🎯 **Agent-Friendly APIs**: RESTful endpoints with x402 protocol integration

**Proof Generation & Verification**: Complete cryptographic validation in 1-8 minutes

**Designed For**:
- Batch settlement and overnight processing workflows
- High-value transactions requiring verifiable authorization ($100,000+)
- Compliance and regulatory reporting with auditable cryptographic proofs
- Agent-to-agent commerce without trusted intermediaries
- Privacy-sensitive scenarios where data exposure creates strategic risk
- Scheduled authorization with flexible timing requirements

**Developer Experience**:
- Policy simulation endpoints for instant testing (<1ms)
- Webhook support for async proof generation
- Comprehensive SDK with TypeScript/Python clients
- 14 ready-to-use authorization models covering common use cases

## ✨ Features

- ✅ **ONNX Inference Proofs**: Zero-knowledge proofs of correct ML model execution
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
curl https://zk-x402.com/.well-known/x402

# 2. Request authorization (receive 402 Payment Required)
curl -X POST https://zk-x402.com/x402/authorize/simple_threshold

# 3. Send USDC payment on Base (using ethers.js or web3.js)
# Transaction to: 0x1f409E94684804e5158561090Ced8941B47B0CC6

# 4. Generate zkML proof locally

# 5. Submit payment + proof via X-PAYMENT header
curl -X POST https://zk-x402.com/x402/authorize/simple_threshold \
  -H "X-PAYMENT: <base64-encoded-payment-and-proof>"

# 6. Receive authorization result with X-PAYMENT-RESPONSE header
```

**For local development**: Replace `https://zk-x402.com` with `http://localhost:3001`

**Complete integration guide**: [PAYMENT_GUIDE.md](zkx402-agent-auth/PAYMENT_GUIDE.md)

## 📊 Performance & Use Case Characteristics

### Zero-Knowledge Proof Generation & Verification Times

zkX402 provides **cryptographically verifiable privacy** through zero-knowledge proofs. Each authorization involves comprehensive cryptographic validation:

| Model Type | Proof Generation | Verification | **Total Time** | Proof Size |
|------------|------------------|--------------|----------------|------------|
| **Simple Policies** | 5-7s | 1-6 minutes | **1-6.5 minutes** | 15-18 KB |
| **Velocity Control** | 6-8s | 40s-4 minutes | **1-5 minutes** | 16-19 KB |
| **Advanced/Neural** | 8-10s | 4-7.5 minutes | **5-8 minutes** | 19-25 KB |

*Measured on: Intel i7, 16GB RAM. Times include complete cryptographic validation.*

### What Happens During These Minutes

**Proof Generation Phase** (5-10 seconds):
1. **Trace Execution** (~1-2s): Record complete ONNX model execution
2. **Constraint Building** (~1-2s): Generate R1CS constraints proving correctness
3. **Polynomial Commitments** (~2-3s): Create cryptographic commitments (Dory scheme)
4. **Spartan Proving** (~2-4s): Generate SNARK proof for constraint satisfaction

**Verification Phase** (40 seconds - 7.5 minutes):
1. **Spartan R1CS Verification**: Comprehensive sumcheck protocol validation
2. **Tensor Heap Verification**: Verify memory consistency across enhanced operations
3. **Instruction Lookup Verification**: Validate extended operation set integrity
4. **Bytecode Verification**: Validate execution trace correctness
5. **Precompile Verification**: Verify specialized operations (MatMul, Gather, etc.)

**Result**: Complete cryptographic guarantee that authorization logic executed correctly, without revealing sensitive data (balances, transaction history, account details).

**What You Get**: The enhanced JOLT Atlas fork (Gather operations, Div, Cast, MAX_TENSOR_SIZE 64→1024) enables sophisticated authorization models. The comprehensive verification ensures these enhanced capabilities are cryptographically sound.

### Ideal Use Cases

zkX402's thorough cryptographic validation (1-8 minutes) makes it particularly well-suited for scenarios where **verifiable privacy and trustless verification provide strategic value**:

| Use Case | Why zkX402 Excels |
|----------|-------------------|
| **Overnight Batch Settlement** | Process thousands of authorizations with cryptographic proofs during off-peak hours |
| **Compliance & Regulatory Reporting** | Generate audit-ready cryptographic proofs for SOX, AML, PCI-DSS submissions |
| **High-Value Escrow Transactions** ($100,000+) | Cryptographic verification for massive wire transfers and settlements |
| **Quarterly Financial Close** | End-of-period authorization verification with verifiable audit trail |
| **Agent-to-Agent Commerce** | Trustless authorization between autonomous agents operating asynchronously |
| **Smart Contract Settlement** | Pre-compute proofs offline, submit verifiable proofs on-chain for settlement |
| **Privacy-Critical Forensics** | Prove authorization decisions without exposing sensitive financial data |
| **Scheduled Procurement** | Automated purchasing systems with flexible timing and verifiable policy compliance |
| **Research & Academic Use** | Privacy-preserving ML model validation and zkML technology demonstration |
| **Regulatory Audit Trail** | Maintain cryptographically verifiable authorization history for compliance |

### Integration Patterns

**Pattern 1: Async Job Queue (Recommended for Production)**
```javascript
// Enqueue proof generation, receive immediate job ID
const { job_id } = await axios.post('/api/proof-jobs', {
  model: 'simple_threshold',
  inputs: { amount: 5000, balance: 10000 },
  callback_url: 'https://your-system.com/proof-ready'
});

console.log('Proof job queued:', job_id);
// Webhook notification arrives in 1-8 minutes with completed proof
```

**Pattern 2: Scheduled Batch Processing**
```bash
# Cron job: Generate proofs overnight
0 2 * * * /usr/bin/generate-daily-proofs.sh

# Proofs available in morning for review and submission
```

**Pattern 3: Pre-Computation for Predictable Workflows**
```javascript
// Anticipate authorization needs, generate proofs in advance
const upcomingTransactions = getScheduledTransactions();
for (const tx of upcomingTransactions) {
  generateProofAsync(tx); // Completes in 1-8 minutes per proof
}

// When authorization needed, proof is ready for immediate use
```

### Client Configuration for Async Operations

Configure timeouts to accommodate full cryptographic verification:

```python
# Python - 10 minute timeout for complete validation
response = requests.post(
    'https://zk-x402.com/api/generate-proof',
    json={'model': 'simple_threshold', 'inputs': {...}},
    timeout=600  # 10 minutes
)
```

```javascript
// JavaScript - 10 minute timeout
const response = await axios.post(
  'https://zk-x402.com/api/generate-proof',
  { model: 'simple_threshold', inputs: {...} },
  { timeout: 600000 }  # 10 minutes in milliseconds
);
```

```rust
// Rust - 10 minute timeout
let client = reqwest::Client::builder()
    .timeout(Duration::from_secs(600))
    .build()?;
```

### Endpoint Timing Guidelines

| Endpoint | Expected Time | Recommended Timeout |
|----------|--------------|---------------------|
| `/.well-known/x402` | < 50ms | 5s |
| `/api/policies` | < 50ms | 5s |
| `/api/policies/:id/schema` | < 50ms | 5s |
| `/api/policies/:id/simulate` | < 1ms | 5s |
| `/api/generate-proof` | 1-8 minutes | 10 minutes |
| `/x402/authorize/:modelId` | 1-8 minutes | 10 minutes |

### Fast Testing with Policy Simulation

For rapid development and testing without generating cryptographic proofs:

```javascript
// Instant simulation (< 1ms, free)
const simulation = await axios.post('/api/policies/simple_threshold/simulate', {
  inputs: { amount: 5000, balance: 10000 }
});

console.log('Instant result:', simulation.data.approved);
// Use for: Testing, validation, development, low-stakes decisions
```

### Comparison: zkML Proofs vs Traditional Authorization

| Characteristic | zkX402 (zkML Proofs) | Traditional Auth |
|----------------|----------------------|------------------|
| **Authorization Time** | 1-8 minutes | <1 millisecond |
| **Privacy Guarantee** | Cryptographic (zero-knowledge) | None (trust-based) |
| **Verifiability** | Anyone can verify cryptographically | Must trust authorizer |
| **Data Exposure** | Zero (balance/history hidden) | Full (server sees all data) |
| **Tamper Resistance** | Cryptographically impossible to fake | Depends on access controls |
| **Audit Trail** | Cryptographic proof preserved forever | Logs (can be modified) |
| **Best For** | High-value, privacy-critical, compliance | High-frequency, interactive |

**Selection Criteria**: Choose zkX402 when cryptographic verifiability and privacy provide value that justifies comprehensive cryptographic validation.

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

# Build Rust prover binary (recommended for performance)
cd jolt-atlas-fork/zkml-jolt-core
cargo build --release --example proof_json_output
cd ../..

# Start web UI
cd ui
npm install
npm run dev
```

Open http://localhost:3000 in your browser (or https://zk-x402.com for production).

### 5-Minute Demo

**1. Generate a Proof (Web UI)**

Navigate to https://zk-x402.com (or http://localhost:3000 for local), select "Simple Threshold" model, and click "Generate Proof" with default inputs:
- Amount: 50 ($0.50)
- Balance: 1000 ($10.00)

Result: **✅ APPROVED** with cryptographic proof (completes in 1-6.5 minutes)

**2. Generate a Proof (API)**

```bash
# Production
curl -X POST https://zk-x402.com/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": "5000",
      "balance": "10000"
    }
  }'

# Note: Response arrives after 1-6.5 minutes with complete cryptographic proof
```

**3. Instant Simulation (for Testing)**

```bash
# Instant response (< 1ms) without cryptographic proof
curl -X POST https://zk-x402.com/api/policies/simple_threshold/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "amount": 5000,
      "balance": 10000
    }
  }'
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

| Model | What It Does | When To Use | Proof Time |
|-------|-------------|-------------|------------|
| **simple_threshold** | Checks if you have enough money | Basic wallet balance checks in x402 payments | 1-6.5 min |
| **percentage_limit** | Limits spending to X% of balance | "Don't spend more than 10% at once" policies | 1-6 min |
| **vendor_trust** | Requires minimum vendor reputation | x402 marketplace transactions | 1-6 min |
| **velocity_1h** | Limits spending per hour | Rate limiting, fraud prevention for x402 agents | 1-5 min |
| **velocity_24h** | Limits spending per day | Daily spending caps for x402 payments | 1-5 min |
| **daily_limit** | Hard cap on daily spending | Budget enforcement for agent spending | 1-5 min |
| **age_gate** | Checks minimum age | Age-restricted x402 purchases | 1-6 min |
| **multi_factor** | Combines balance + velocity + trust | High-security x402 transactions | 5-8 min |
| **composite_scoring** | Weighted risk score from multiple factors | Advanced risk assessment for x402 | 5-8 min |
| **risk_neural** | ML-based risk scoring | Sophisticated fraud detection for agent payments | 5-8 min |

#### Test Models (4)

| Model | What It Does | When To Use | Proof Time |
|-------|-------------|-------------|------------|
| **test_less** | Tests Less comparison operation | Operation testing, JOLT Atlas validation | 1-5 min |
| **test_identity** | Tests Identity pass-through operation | Graph construction, residual connections | 1-5 min |
| **test_clip** | Tests Clip operation (ReLU approximation) | Activation function testing | 1-5 min |
| **test_slice** | Tests Slice tensor operation | Feature extraction, tensor manipulation | 1-5 min |

### Quick Start with Curated Models

```bash
# 1. Test a model with sample inputs
cd zkx402-agent-auth/policy-examples/onnx/curated
python3 test_all_models.py

# 2. Generate proof for specific model (completes in 1-8 minutes)
cd ../../jolt-atlas-fork/zkml-jolt-core
./target/release/examples/proof_json_output \
  ../../policy-examples/onnx/curated/simple_threshold.onnx \
  5000 10000  # $50 and $100 in cents
```

### Real-World x402 Example

**Scenario**: An AI shopping agent needs to buy office supplies ($75) via x402 in a scheduled procurement workflow.

**Without zkML**:
- Agent reveals balance to vendor: "I have $10,000, so $75 is fine"
- Privacy compromised, creates security risk in x402 payment flow

**With zkML (using simple_threshold.onnx)**:
- Agent generates proof overnight: "I ran the authorization model and it approved"
- Vendor verifies proof in x402 flow next day: Valid ✓
- Balance stays private, authorization confirmed cryptographically
- Proof generated during off-peak hours (1-6 minutes)

### Model Details

All 14 models are:
- ✅ **Validated**: Pass comprehensive test suites (29 test cases)
- ✅ **JOLT-Compatible**: Use only supported operations (Add, Sub, Mul, Div, Greater, Less, Cast, Clip, Identity, Slice, Gather)
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
- **[AGENT_INTEGRATION.md](zkx402-agent-auth/AGENT_INTEGRATION.md)**: AI agent integration patterns
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
- **Gather Operations**: Enhanced index-based tensor access for flexible input handling
- **MatMult Enhancements**: Extended support for 1D tensor outputs in neural network policies
- **Increased Tensor Size Limits**: `MAX_TENSOR_SIZE` increased from 64→1024 to support larger authorization models

**What This Enables for x402**:
- ✅ **Verifiable Payment Authorization**: Cryptographically prove spending limit checks executed correctly
- ✅ **Trustless Rule-Based Policies**: Verifiable threshold checks for x402 transactions
- ✅ **Auditable Neural Network Decisions**: Provably demonstrate ML risk scoring was computed correctly
- ✅ **Tamper-Proof Velocity Controls**: Cryptographic proof that rate limiting was enforced faithfully
- ✅ **Transparent Multi-Criteria Logic**: Verifiable complex authorization with zero-knowledge proofs

**Core Innovation**: These enhancements enable **verifiable ONNX inference** with comprehensive cryptographic validation - proofs that an ONNX model was executed correctly with specific inputs, producing a specific output. The x402 ecosystem can verify authorization decisions cryptographically rather than trusting agent claims, making authorization **trustless** and **auditable**.

See [JOLT_ATLAS_ENHANCEMENTS.md](zkx402-agent-auth/jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md) for technical details.

### 2. Web UI

Interactive interface for testing x402 authorization policies:

- **Authorization Simulator**: Test x402 payment policies with custom inputs
- **Model Comparison**: Compare authorization results across different policy models
- **Proof History**: View and export previous authorization proofs
- **Model Registry**: Upload and manage custom ONNX authorization models for x402
- **Real-Time Progress**: Animated loading states with proof generation stages

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
GET  /.well-known/x402         - x402 service discovery
GET  /api/models               - List authorization models
GET  /api/policies             - Agent-readable policy catalog
POST /api/policies/:id/simulate - Instant simulation (<1ms)
POST /api/generate-proof       - Generate cryptographic proof (1-8 minutes)
POST /x402/authorize/:modelId  - x402 payment-gated authorization
```

**Features**:
- Rate limiting (5 free proofs/day for UI, unlimited for x402 payments)
- Request ID tracking for audit trails
- Structured logging for x402 transaction monitoring
- Input validation for payment authorization requests
- Webhook support for async proof notifications
- CORS support for web-based x402 applications

**Start API**:
```bash
cd zkx402-agent-auth/ui
npm install
npm start
```

### 4. Model Registry

Upload and manage custom x402 authorization models:

```bash
# Via UI: Drag & drop .onnx files for custom x402 payment policies

# Via API (Production):
curl -X POST https://zk-x402.com/api/upload-model \
  -F "model=@my_x402_policy.onnx" \
  -F "description=Custom x402 payment authorization policy"

# Local development: http://localhost:3001/api/upload-model
```

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
cd zkx402-agent-auth/jolt-atlas-fork/zkml-jolt-core
cargo test

# Model validation
cd zkx402-agent-auth/policy-examples/onnx/curated
python3 test_all_models.py
```

### Test Coverage

- ✅ Proof generation (approved/rejected cases)
- ✅ Model validation (all 14 models)
- ✅ API endpoints
- ✅ Batch processing
- ✅ Error handling
- ✅ Cryptographic verification

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
    ├── policy-examples/onnx/           # Model generation
    │   ├── curated/                   # 14 production models
    │   ├── create_demo_models.py      # Generate demos
    │   └── test_models.py             # Validate models
    ├── ui/                             # React web interface
    │   ├── src/components/
    │   │   ├── AuthorizationSimulator.jsx
    │   │   ├── ProofHistory.jsx       # Proof history
    │   │   ├── ModelComparison.jsx    # Model comparison
    │   │   └── LoadingIndicator.jsx   # Progress animation
    │   ├── tests/                     # Jest tests
    │   └── server.js                  # Express API
    └── sdk/python/                     # Python SDK
        ├── zkx402_client.py            # Sync client
        └── zkx402_async_client.py      # Async client
```

## 🌐 x402 Integration Examples

### JavaScript/TypeScript - Async x402 Payment Flow

```javascript
const axios = require('axios');

/**
 * Generate verifiable inference proof for x402 payment authorization
 * Uses async pattern for optimal UX
 */
async function authorizeX402PaymentAsync(amount, balance, trust) {
  // Initiate proof generation (returns immediately)
  const response = await axios.post('https://zk-x402.com/api/generate-proof', {
    model: 'simple_threshold',
    inputs: {
      amount: amount.toString(),
      balance: balance.toString(),
      vendor_trust: trust.toString()
    },
    webhook_id: 'your-webhook-endpoint'  // Receive notification when ready
  });

  const { request_id } = response.data;
  console.log('Proof generation initiated:', request_id);

  // Proof arrives via webhook in 1-8 minutes
  return { request_id, status: 'processing' };
}

// Webhook handler receives completed proof
app.post('/webhook', (req, res) => {
  const { request_id, approved, zkmlProof } = req.body;
  console.log('x402 Payment Authorized:', approved);
  console.log('Cryptographic proof ready:', zkmlProof.commitment);
  // Process authorization result
});
```

### Python - Scheduled Batch Authorization

```python
import requests
import time

def batch_authorize_x402_payments(transactions):
    """
    Generate verifiable proofs for batch of transactions.
    Ideal for overnight processing or scheduled workflows.
    """
    proof_jobs = []

    # Submit all proof generation jobs
    for tx in transactions:
        response = requests.post('https://zk-x402.com/api/generate-proof', json={
            'model': 'simple_threshold',
            'inputs': {
                'amount': str(tx['amount']),
                'balance': str(tx['balance'])
            },
            'webhook_id': f'batch-{tx["id"]}'
        })
        proof_jobs.append(response.json()['request_id'])

    print(f'Submitted {len(proof_jobs)} proof generation jobs')
    print('Proofs will be ready in 1-8 minutes each')

    return proof_jobs

# Usage in scheduled job
transactions = get_pending_authorizations()
job_ids = batch_authorize_x402_payments(transactions)
print(f'Batch processing initiated: {len(job_ids)} authorizations')
```

### Rust - Native x402 Integration

```rust
use zkml_jolt_core::jolt::*;
use onnx_tracer::{model, tensor::Tensor};

/// Generate verifiable ONNX inference proof for x402 payment authorization
/// Completes in 1-8 minutes depending on model complexity
fn authorize_x402_payment(amount: i32, balance: i32, trust: i32) -> Result<(bool, JoltProof)> {
    let model_obj = model(&"simple_threshold.onnx".into());
    let inputs = vec![amount, balance, trust];
    let input_tensor = Tensor::new(Some(&inputs), &vec![1, 3])?;

    // Generate and verify proof via JOLT Atlas (1-8 minutes)
    let (proof, output) = generate_and_verify_proof(model_obj, input_tensor)?;

    // Authorization decision from model inference
    let approved = output.inner[0] > 50;

    Ok((approved, proof))
}

// Usage in async workflow
let (approved, proof) = authorize_x402_payment(50, 1000, 80)?;
println!("x402 Payment Authorized: {}", approved);
println!("Cryptographic Proof Generated: {} bytes", proof.size());
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

# Build Rust prover
cd zkx402-agent-auth/jolt-atlas-fork/zkml-jolt-core
cargo build --release --example proof_json_output

# Build and run API
cd ../../ui
npm install
npm run build
npm start
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete instructions.

## 📈 Roadmap

### ✅ Completed
- [x] Enhanced JOLT Atlas with comparison operations and increased tensor sizes
- [x] ONNX inference proof generation with comprehensive verification
- [x] Rule-based authorization policies for x402
- [x] Neural network authorization policies for x402
- [x] Web UI with proof history
- [x] REST API for x402 service integration
- [x] Base L2 USDC payment integration
- [x] Model registry and upload
- [x] Comprehensive test suite
- [x] Production deployment guides
- [x] Async webhook support for proof notifications

### 🚧 In Progress
- [ ] x402 protocol integration examples
- [ ] x402 Bazaar marketplace integration
- [ ] Multi-agent x402 commerce demonstrations

### 🔮 Future
- [ ] Proof caching by input hash for repeated authorizations
- [ ] WebSocket support for real-time proof status updates
- [ ] Model training pipeline in UI for custom x402 policies
- [ ] GPU acceleration for proof generation
- [ ] Batch proof aggregation for improved throughput

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
