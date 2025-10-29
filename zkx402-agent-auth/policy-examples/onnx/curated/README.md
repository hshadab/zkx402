# Curated Authorization Models

14 curated ONNX models (10 production + 4 test) for x402 agent authorization with zkX402.

## Quick Start

```bash
# Test a model with JOLT prover
cd jolt-prover
./target/release/examples/proof_json_output \
  ../policy-examples/onnx/curated/simple_threshold.onnx \
  5000 10000  # $50 amount, $100 balance
```

## Available Models

### Production Models (10)

| Model | Use Case | Speed | Inputs |
|-------|----------|-------|--------|
| `simple_threshold` | Basic balance check | ⚡⚡⚡ | amount, balance |
| `percentage_limit` | Max % of balance | ⚡⚡⚡ | amount, balance, max_percentage |
| `vendor_trust` | Reputation check | ⚡⚡⚡ | vendor_trust, min_trust |
| `velocity_1h` | 1-hour rate limit | ⚡⚡ | amount, spent_1h, limit_1h |
| `velocity_24h` | 24-hour rate limit | ⚡⚡ | amount, spent_24h, limit_24h |
| `daily_limit` | Daily spending cap | ⚡⚡ | amount, daily_spent, daily_cap |
| `age_gate` | Age verification | ⚡⚡⚡ | age, min_age |
| `multi_factor` | Comprehensive check | ⚡ | 6 inputs (balance+velocity+trust) |
| `composite_scoring` | Weighted risk score | ⚡⚡ | amount, balance, vendor_trust, user_history |
| `risk_neural` | ML-based risk | ⚡ | amount, balance, velocity_1h, velocity_24h, vendor_trust |

### Test Models (4)

| Model | Use Case | Speed | Inputs |
|-------|----------|-------|--------|
| `test_less` | Less comparison testing | ⚡⚡⚡ | value1, value2 |
| `test_identity` | Identity operation testing | ⚡⚡⚡ | input |
| `test_clip` | Clip/ReLU testing | ⚡⚡⚡ | input |
| `test_slice` | Slice operation testing | ⚡⚡⚡ | tensor, start, end |

## API Usage

```javascript
// Use with zkX402 API
const response = await fetch('https://api.zkx402.ai/generate-proof', {
  method: 'POST',
  body: JSON.stringify({
    model: 'multi_factor',  // Built-in model
    inputs: {
      amount: 5000,
      balance: 100000,
      spent_24h: 20000,
      limit_24h: 50000,
      vendor_trust: 75,
      min_trust: 50
    }
  })
});

const { approved, zkmlProof } = await response.json();
```

## Model Selection

**For basic wallet checks**: `simple_threshold`
**For marketplace**: `vendor_trust` or `multi_factor`
**For fraud prevention**: `velocity_1h`, `velocity_24h`
**For age-restricted goods**: `age_gate`
**For maximum security**: `multi_factor` or `risk_neural`

## Documentation

See [CATALOG.md](./CATALOG.md) for complete specifications, test cases, and implementation details.

## Regenerating Models

```bash
cd policy-examples/onnx
python3 curated/generate_all_models.py
python3 curated/generate_risk_neural.py
```

All models are validated for JOLT Atlas compatibility:
- ✅ Operations: Add, Sub, Mul, Div, Greater, Less, Cast, Clip, Identity, Slice
- ✅ Size: < MAX_TENSOR_SIZE (1024 elements)
- ✅ Integer arithmetic with scale factors

---

**Version**: 1.0.0
**Last Updated**: 2025-10-28
**License**: MIT
