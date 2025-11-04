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

Expected output (completes in 1-8 minutes):
```
[1/5] Loading ONNX model...
[2/5] Preprocessing JOLT prover...
[3/5] Preparing authorization inputs...
[4/5] Generating JOLT Atlas proof...
      ✓ Proof generated (5-10 seconds)
[5/5] Verifying proof cryptographically...
      ✓ Proof verified! (40 seconds - 7.5 minutes)

║  ✅ Transaction AUTHORIZED via ZK proof
```

**Note**: Full cryptographic verification takes 1-8 minutes depending on model complexity. This ensures complete verification of the zero-knowledge proof.

### 4. Start Web UI

```bash
cd ../ui
npm install
npm run dev
```

Open http://localhost:3000 in your browser (or https://zk-x402.com for production).

## Quick Test

### Via Web UI

1. Open https://zk-x402.com (or http://localhost:3000 for local development)
2. Select "Simple Auth" model
3. Use default values:
   - Amount: 50 ($0.50)
   - Balance: 1000 ($10.00)
   - Velocity 1h: 20
   - Velocity 24h: 100
   - Vendor Trust: 80
4. Click "Generate Proof"
5. Wait 1-6.5 minutes for cryptographic verification
6. See result: **APPROVED** ✅

### Via API

```bash
# Production
curl -X POST https://zk-x402.com/api/generate-proof \
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

# Local development: http://localhost:3001/api/generate-proof
```

Response (arrives after 1-6.5 minutes):
```json
{
  "approved": true,
  "output": 100,
  "verification": true,
  "proofSize": "15.2 KB",
  "provingTime": "6600ms",
  "verificationTime": "370900ms",
  "operations": 21,
  "zkmlProof": {
    "commitment": "0x...",
    "response": "0x...",
    "evaluation": "0x..."
  }
}
```

**Note**: `verificationTime` includes complete cryptographic validation (40s - 7.5 minutes). This ensures the proof is cryptographically sound.

## Available Models

| Model | Type | Total Time | Description |
|-------|------|------------|-------------|
| `simple_threshold` | Rule-based | 1-6.5 min | Basic threshold checks |
| `risk_neural` | Neural network | 5-8 min | ML-based risk scoring |
| `multi_factor` | Multi-criteria | 5-8 min | Comprehensive authorization |
| `velocity_1h` | Rate limiting | 1-5 min | Hourly spending caps |
| `composite_scoring` | Weighted scoring | 5-8 min | Advanced risk assessment |

**Time includes**: Proof generation (5-10s) + cryptographic verification (40s - 7.5 minutes)

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

## Understanding Proof Times

zkX402 performs comprehensive cryptographic verification to ensure proofs are valid:

- **Proof Generation**: 5-10 seconds (create the zero-knowledge proof)
- **Verification**: 40 seconds - 7.5 minutes (cryptographically validate the proof)
- **Total Time**: 1-8 minutes from request to verified result

**Why verification takes time**: The JOLT Atlas enhancements (Gather, Div, Cast, larger tensors) enable sophisticated authorization policies. Cryptographic verification of these enhanced capabilities requires thorough sumcheck validation.

**Best for**: Batch processing, overnight settlement, compliance reporting, high-value transactions, scheduled workflows.

**For instant testing**: Use `/api/policies/:id/simulate` endpoint (<1ms, no proof generation).

## Next Steps

- **Create Custom Policy**: See `policy-examples/onnx/README.md`
- **Run Tests**: `npm test` (UI), `cargo test` (Rust)
- **Deploy**: See `DEPLOYMENT.md`
- **API Integration**: See `API_REFERENCE.md` for async patterns and webhooks

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
