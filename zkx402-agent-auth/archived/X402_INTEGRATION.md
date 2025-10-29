# x402 Protocol Integration

zkX402 now implements the **Coinbase x402 HTTP payment protocol** for verifiable agent authorization.

## What is x402?

x402 is an HTTP-native payment protocol that extends the 402 Payment Required status code for machine-to-machine payments. [Spec](https://github.com/coinbase/x402)

## zkX402's Unique Approach

Instead of traditional crypto payments, zkX402 uses **zkML proofs as payment credentials**:

1. **Proof = Payment**: The zkML proof itself serves as the authorization payment
2. **Verifiable Authorization**: Agents prove they checked authorization policies correctly
3. **Privacy-Preserving**: No private data revealed, just cryptographic proof of compliance

## x402 Endpoints

### Discovery
```bash
GET /.well-known/x402
```

Returns service metadata, available models, pricing, and x402 capabilities.

**Response**:
```json
{
  "service": "zkX402 Agent Authorization",
  "x402Version": 1,
  "schemes": [{"scheme": "zkml-jolt", "network": "jolt-atlas"}],
  "models": [/* 10 curated models */],
  "endpoints": {
    "models": "/x402/models",
    "authorize": "/x402/authorize/:modelId",
    "verify": "/x402/verify-proof"
  }
}
```

### List Models
```bash
GET /x402/models
```

Lists all 10 curated authorization models with pricing and payment requirements.

### Authorization (402 Flow)
```bash
POST /x402/authorize/:modelId
```

**Without proof** → Returns 402 with payment requirements:
```json
{
  "x402Version": 1,
  "accepts": [{
    "scheme": "zkml-jolt",
    "network": "jolt-atlas",
    "maxAmountRequired": "5000",
    "payTo": "/x402/verify-proof",
    "resource": "/x402/authorize/multi_factor",
    "description": "Multi-Factor: Combines balance + velocity + trust",
    "extra": {
      "modelId": "multi_factor",
      "inputs": ["amount", "balance", "spent_24h", "limit_24h", "vendor_trust", "min_trust"],
      "proofType": "jolt-atlas-onnx"
    }
  }]
}
```

**With proof** (X-PAYMENT header) → Returns 200 with X-PAYMENT-RESPONSE:
```bash
curl -X POST http://localhost:3001/x402/authorize/simple_threshold \
  -H "X-PAYMENT: <base64-encoded-proof>"
```

Response includes `X-PAYMENT-RESPONSE` header with settlement confirmation.

## Payment Flow

```
┌────────────┐                                  ┌────────────┐
│   Agent    │                                  │   zkX402   │
│            │                                  │   Server   │
└────────────┘                                  └────────────┘
      │                                                │
      │  1. POST /x402/authorize/simple_threshold    │
      │ ──────────────────────────────────────────>  │
      │                                               │
      │  2. 402 Payment Required                     │
      │     {accepts: [payment requirements]}        │
      │ <──────────────────────────────────────────  │
      │                                               │
      │  3. Generate zkML proof                      │
      │     (via /api/generate-proof)                │
      │ ──────────────────────────────────────────>  │
      │     {zkmlProof, approved, output}            │
      │ <──────────────────────────────────────────  │
      │                                               │
      │  4. Retry with X-PAYMENT header              │
      │     X-PAYMENT: <base64(proof+metadata)>      │
      │ ──────────────────────────────────────────>  │
      │                                               │
      │  5. 200 OK + X-PAYMENT-RESPONSE              │
      │     {authorized, proof: {verified}}          │
      │ <──────────────────────────────────────────  │
```

## Custom Scheme: `zkml-jolt`

zkX402 defines a custom x402 payment scheme:

- **Scheme**: `zkml-jolt`
- **Network**: `jolt-atlas`
- **Asset**: `zkml-proof` (not a crypto token)
- **Payment**: JOLT Atlas ONNX inference proof

## Model Pricing

| Model | Price (atomic) | Category | Complexity |
|-------|---------------|----------|------------|
| simple_threshold | 1000 | Basic | 2 inputs, ~10 ops |
| percentage_limit | 1500 | Basic | 3 inputs, ~15 ops |
| vendor_trust | 1000 | Basic | 2 inputs, ~5 ops |
| velocity_1h | 2000 | Velocity | 3 inputs, ~10 ops |
| velocity_24h | 2000 | Velocity | 3 inputs, ~10 ops |
| daily_limit | 2000 | Velocity | 3 inputs, ~10 ops |
| age_gate | 1000 | Access | 2 inputs, ~5 ops |
| multi_factor | 5000 | Advanced | 6 inputs, ~30 ops |
| composite_scoring | 4000 | Advanced | 4 inputs, ~25 ops |
| risk_neural | 6000 | Advanced | 5 inputs, ~47 ops |

*Prices reflect computational complexity. Actual payments can be configured.*

## Example: Node.js Client

```javascript
const axios = require('axios');

async function authorizeWithProof(modelId, inputs) {
  // Step 1: Try authorization (will get 402)
  try {
    await axios.post(`http://localhost:3001/x402/authorize/${modelId}`);
  } catch (error) {
    if (error.response?.status === 402) {
      const paymentReq = error.response.data.accepts[0];
      console.log('Payment required:', paymentReq.description);

      // Step 2: Generate zkML proof
      const proofRes = await axios.post('http://localhost:3001/api/generate-proof', {
        model: modelId,
        inputs
      });

      // Step 3: Retry with X-PAYMENT header
      const payment = Buffer.from(JSON.stringify({
        x402Version: 1,
        scheme: 'zkml-jolt',
        network: 'jolt-atlas',
        payload: {
          modelId,
          zkmlProof: proofRes.data
        }
      })).toString('base64');

      const authRes = await axios.post(
        `http://localhost:3001/x402/authorize/${modelId}`,
        {},
        { headers: { 'X-PAYMENT': payment } }
      );

      console.log('Authorized:', authRes.data);
      console.log('Settlement:', authRes.headers['x-payment-response']);
      return authRes.data;
    }
  }
}

// Use it
authorizeWithProof('simple_threshold', {
  amount: '5000',
  balance: '10000'
});
```

## Middleware

The x402 middleware (`x402-middleware.js`) handles:
- Parsing X-PAYMENT headers
- Verifying zkML proofs
- Generating 402 responses
- Encoding X-PAYMENT-RESPONSE headers
- Dynamic model support

## Testing

```bash
# Start server
npm run dev

# Test discovery
curl http://localhost:3001/.well-known/x402

# Test models list
curl http://localhost:3001/x402/models

# Test 402 flow (will return 402)
curl -X POST http://localhost:3001/x402/authorize/simple_threshold

# Generate proof
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{"model":"simple_threshold","inputs":{"amount":"5000","balance":"10000"}}'

# Use proof in X-PAYMENT header (requires base64 encoding)
```

## Key Features

✅ **Full x402 Protocol Compliance**
- 402 status codes
- X-PAYMENT / X-PAYMENT-RESPONSE headers
- Discovery endpoint (/.well-known/x402)
- Payment requirements specification

✅ **zkML-Specific Extensions**
- Custom scheme: `zkml-jolt`
- Proof-based payment
- Model-specific pricing
- Dynamic input validation

✅ **Production Ready**
- All 10 curated models supported
- Proper error handling
- Structured logging
- Health checks

✅ **Developer Friendly**
- Clear API documentation
- Example client code
- Standard HTTP/JSON
- No blockchain knowledge required

## Architecture

```
┌─────────────────────────────────────────────┐
│           x402 Protocol Layer               │
│  /.well-known/x402, 402 responses,          │
│  X-PAYMENT headers, payment requirements    │
├─────────────────────────────────────────────┤
│         zkML Authorization Layer            │
│  Proof generation, verification,            │
│  model selection, input validation          │
├─────────────────────────────────────────────┤
│           JOLT Atlas Prover                 │
│  ONNX inference proofs, cryptographic       │
│  verification, zero-knowledge guarantees    │
└─────────────────────────────────────────────┘
```

## Why x402 for zkML?

1. **HTTP-Native**: Works with existing HTTP infrastructure
2. **Machine-to-Machine**: Perfect for agent authorization
3. **Minimal Integration**: Single middleware, standard headers
4. **Extensible**: Custom schemes support zkML proofs
5. **Discovery**: Agents can find and select authorization models
6. **Verifiable**: Cryptographic proofs replace trust

## Future Enhancements

- [ ] Actual crypto settlement integration
- [ ] Facilitator service for on-chain verification
- [ ] Batch proof verification
- [ ] Proof caching and reuse
- [ ] Rate limiting per model
- [ ] Usage analytics

---

**Spec**: https://github.com/coinbase/x402
**Implementation**: `/ui/server.js` + `/ui/x402-middleware.js`
**Date**: 2025-10-28
