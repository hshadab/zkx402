# zkX402: Zero-Knowledge Agent Authorization

Privacy-preserving authorization for AI agents using JOLT Atlas zkML proofs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![JOLT Atlas](https://img.shields.io/badge/JOLT-Atlas-blue.svg)](https://github.com/ICME-Lab/jolt-atlas)

## 🚀 What is zkX402?

zkX402 enables AI agents to **prove they're authorized to perform actions without revealing private data**. Using zero-knowledge machine learning proofs, agents can demonstrate compliance with spending policies while keeping balances, transaction history, and velocity metrics completely private.

**Use Cases**:
- 💰 Agent spending authorization (prove `amount < budget` without revealing budget)
- 🔐 Access control (prove eligibility without revealing credentials)
- 📊 Risk assessment (prove low-risk transaction without revealing financial history)
- 🤝 Compliance verification (prove policy adherence without exposing data)

## ✨ Features

- ✅ **Real Zero-Knowledge Proofs**: JOLT Atlas-based cryptographic proofs (~0.7s generation)
- ✅ **Rule-Based Policies**: Threshold checks, comparisons, velocity limits
- ✅ **Neural Network Policies**: ML-based risk scoring and classification
- ✅ **Web UI**: Interactive proof generation and model comparison
- ✅ **REST API**: Production-ready external API with rate limiting
- ✅ **Model Registry**: Upload and manage custom ONNX authorization models
- ✅ **Proof History**: Persistent history with export functionality
- ✅ **Batch Processing**: Generate multiple proofs in parallel
- ✅ **Comprehensive Tests**: Full E2E test coverage (Jest + Rust)

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

### 1. Enhanced JOLT Atlas Prover

We've extended JOLT Atlas to support practical authorization use cases:

**Additions Made**:
- **Comparison Operations**: `Greater`, `Less`, `GreaterEqual` for threshold checks
- **Tensor Operations**: `Slice`, `Identity`, `Reshape` for data manipulation
- **MatMult Enhancements**: Extended support for 1D tensor outputs

**What This Enables**:
- ✅ Rule-based policies with thresholds (amount < 10% of balance)
- ✅ Neural network authorization models
- ✅ Velocity checks and rate limiting
- ✅ Multi-criteria authorization decisions

See [JOLT_ATLAS_ENHANCEMENTS.md](zkx402-agent-auth/jolt-atlas-fork/JOLT_ATLAS_ENHANCEMENTS.md) for technical details.

### 2. Web UI

Interactive interface for proof generation and model management:

- **Authorization Simulator**: Test policies with custom inputs
- **Model Comparison**: Compare authorization results across models
- **Proof History**: View and export previous proofs
- **Model Registry**: Upload and manage custom ONNX models
- **Real-Time Progress**: Animated loading states with stage tracking

**Start UI**:
```bash
cd zkx402-agent-auth/ui
npm install
npm run dev
```

### 3. REST API Server

Production-ready API for external integration:

**Endpoints**:
```
GET  /api/v1/health            - Health check
GET  /api/v1/models            - List models
POST /api/v1/proof             - Generate single proof
POST /api/v1/proof/batch       - Batch proof generation
```

**Features**:
- Rate limiting (100 req/15min)
- Request ID tracking
- Structured logging
- Input validation
- CORS support

**Start API**:
```bash
cd zkx402-agent-auth/api-server
npm install
npm start
```

### 4. Model Registry

Upload and manage custom authorization models:

```bash
# Via UI: Drag & drop .onnx files

# Via API:
curl -X POST http://localhost:3001/api/upload-model \
  -F "model=@my_policy.onnx" \
  -F "description=Custom authorization policy"
```

## 📊 Performance

| Model Type | Proving Time | Verification | Proof Size | Operations |
|------------|--------------|--------------|------------|-----------|
| Simple Auth | 0.7s | 45ms | 15 KB | 21 |
| Neural Network | 1.5s | 65ms | 40 KB | 45 |
| Complex NN | 3.0s | 100ms | 80 KB | 95 |

*Measured on: Intel i7, 16GB RAM*

**Optimizations**:
- Rust release builds with LTO
- Dory polynomial commitment scheme
- Integer-only arithmetic (no floating point)
- Efficient ONNX operation tracing

## 🔐 Authorization Policies

### Rule-Based Policy

```python
import torch

class RuleBasedAuth(torch.nn.Module):
    def forward(self, amount, balance, velocity_1h, velocity_24h, trust):
        # Scale to integers (100 = 1.00)
        amount_i = (amount * 100).int()
        balance_i = (balance * 100).int()

        # Rules
        rule1 = amount_i < (balance_i * 10 // 100)  # amount < 10% balance
        rule2 = (trust * 100).int() > 50             # trust > 0.5
        rule3 = (velocity_1h * 100).int() < (balance_i * 5 // 100)

        # All must pass
        approved = (rule1 & rule2 & rule3).int()
        return approved

# Export to ONNX
torch.onnx.export(model, inputs, "rule_based.onnx", opset_version=14)
```

### Neural Network Policy

```python
class NeuralAuth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8)
        self.fc2 = torch.nn.Linear(8, 4)
        self.fc3 = torch.nn.Linear(4, 1)

    def forward(self, amount, balance, velocity_1h, velocity_24h, trust):
        # Stack inputs (already scaled to integers)
        x = torch.stack([amount, balance, velocity_1h, velocity_24h, trust])

        # Neural network
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        score = self.fc3(x)

        # Threshold
        approved = (score > 0.5).int()
        return approved
```

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

## 🌐 Integration Examples

### JavaScript/TypeScript

```javascript
const axios = require('axios');

async function authorizeTransaction(amount, balance, trust) {
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

  return response.data.approved;
}

// Usage
const approved = await authorizeTransaction(50, 1000, 80);
console.log('Authorized:', approved);
```

### Python

```python
import requests

def authorize_transaction(amount, balance, trust):
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

    return response.json()['approved']

# Usage
approved = authorize_transaction(50, 1000, 80)
print(f'Authorized: {approved}')
```

### Rust

```rust
use zkml_jolt_core::jolt::*;
use onnx_tracer::{model, tensor::Tensor};

fn authorize_transaction(amount: i32, balance: i32, trust: i32) -> Result<bool> {
    let model_obj = model(&"simple_auth.onnx".into());
    let inputs = vec![amount, balance, 20, 100, trust];
    let input_tensor = Tensor::new(Some(&inputs), &vec![1, 5])?;

    let (proof, output) = generate_and_verify_proof(model_obj, input_tensor)?;

    Ok(output.inner[0] > 50)
}

// Usage
let approved = authorize_transaction(50, 1000, 80)?;
println!("Authorized: {}", approved);
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

- [x] Enhanced JOLT Atlas with comparison operations
- [x] Rule-based authorization policies
- [x] Neural network authorization policies
- [x] Web UI with proof history
- [x] REST API for external integration
- [x] Model registry and upload
- [x] Comprehensive test suite
- [x] Production deployment guides
- [ ] N-API bindings for faster proof generation
- [ ] Proof caching by input hash
- [ ] WebSocket support for real-time updates
- [ ] Standalone verifier application
- [ ] Model training pipeline in UI
- [ ] Multi-tenant support

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

- **JOLT Atlas Team**: Original zkML framework
- **ONNX Community**: Model format and tooling
- **Rust Crypto Community**: Cryptographic primitives

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/hshadab/zkx402/issues)
- **Documentation**: [Full Docs](https://github.com/hshadab/zkx402/tree/main/docs)
- **JOLT Atlas**: [Original Project](https://github.com/ICME-Lab/jolt-atlas)

---

**Built with** ❤️ **using JOLT Atlas zero-knowledge machine learning**

**Status**: 🚀 Production Ready
