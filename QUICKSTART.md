# zkX402 Quickstart Guide

Get started with zkX402 JOLT Atlas agent authorization in 5 minutes.

## Prerequisites

- **Rust**: 1.70+ ([Install Rust](https://rustup.rs/))
- **Node.js**: 20+ ([Install Node.js](https://nodejs.org/))
- **Python**: 3.8+ (for ONNX model generation)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/zkx402.git
cd zkx402/zkx402-agent-auth
```

### 2. Generate ONNX Models

```bash
cd policy-examples/onnx
python3 create_demo_models.py
cd ../..
```

You should see:
```
✓ Created simple_auth.onnx
✓ Created neural_auth.onnx
✓ Created comparison_demo.onnx
✓ Created tensor_ops_demo.onnx
✓ Created matmult_1d_demo.onnx
```

### 3. Test JOLT Atlas Proof Generation

```bash
cd jolt-prover
cargo run --release --example integer_auth_e2e
```

Expected output:
```
[1/5] Loading ONNX model...
[2/5] Preprocessing JOLT prover...
[3/5] Preparing authorization inputs...
[4/5] Generating JOLT Atlas proof...
      ✓ Proof generated in ~700ms
[5/5] Verifying proof...
      ✓ Proof verified!

║  ✅ Transaction AUTHORIZED via ZK proof
```

### 4. Start Web UI

```bash
cd ../ui
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Quick Test

### Via Web UI

1. Open http://localhost:3000
2. Select "Simple Auth" model
3. Use default values:
   - Amount: 50 ($0.50)
   - Balance: 1000 ($10.00)
   - Velocity 1h: 20
   - Velocity 24h: 100
   - Vendor Trust: 80
4. Click "Generate Proof"
5. See result: **APPROVED** ✅

### Via API

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

Response:
```json
{
  "approved": true,
  "output": 100,
  "verification": true,
  "proofSize": "15.2 KB",
  "verificationTime": "45ms",
  "operations": 21,
  "zkmlProof": {
    "commitment": "0x...",
    "response": "0x...",
    "evaluation": "0x..."
  }
}
```

## Available Models

| Model | Type | Proving Time | Description |
|-------|------|--------------|-------------|
| `simple_auth` | Rule-based | ~0.7s | Basic threshold checks |
| `neural_auth` | Neural network | ~1.5s | ML-based risk scoring |
| `comparison_demo` | Demo | ~0.3s | Comparison operations showcase |
| `tensor_ops_demo` | Demo | ~0.3s | Tensor operations showcase |
| `matmult_1d_demo` | Demo | ~0.4s | Matrix multiplication demo |

## Authorization Logic

### Simple Auth Rules

Transaction approved if **ALL** conditions met:
1. `amount < 10% of balance`
2. `vendor_trust > 0.5`
3. `velocity_1h < 5% of balance`
4. `velocity_24h < 20% of balance`

### Neural Auth

ML model trained on transaction patterns:
- Input: [amount, balance, velocity_1h, velocity_24h, vendor_trust]
- Architecture: [5] → [8 hidden] → [4 hidden] → [1 output]
- Output > 0.5 = APPROVED

## Next Steps

- **Create Custom Policy**: See `policy-examples/onnx/README.md`
- **Run Tests**: `npm test` (UI), `cargo test` (Rust)
- **Deploy**: See `DEPLOYMENT.md`
- **API Integration**: See `API_REFERENCE.md`

## Troubleshooting

### Model Not Found

```bash
cd zkx402-agent-auth/policy-examples/onnx
python3 create_demo_models.py
```

### Rust Build Fails

```bash
rustup update
cargo clean
cargo build --release
```

### UI Won't Start

```bash
cd zkx402-agent-auth/ui
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Support

- **Issues**: https://github.com/yourusername/zkx402/issues
- **Documentation**: See README.md and docs/
- **JOLT Atlas**: https://github.com/ICME-Lab/jolt-atlas

---

**Next**: Read [API_REFERENCE.md](./API_REFERENCE.md) for API integration details.
