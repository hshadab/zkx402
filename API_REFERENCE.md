# zkX402 API Reference

Complete API documentation for zkX402 Agent Authorization service.

## Base URL

### Production
```
https://zk-x402.com/api
```

### Local Development
```
http://localhost:3001/api
```

## Authentication

Currently no authentication required. In production, implement your preferred auth method (JWT, API keys, etc.).

## Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-10-28T12:00:00.000Z",
  "modelsDir": "/path/to/models",
  "modelsAvailable": 5
}
```

**Status Codes:**
- `200`: Service healthy
- `500`: Service unhealthy

---

### GET /models

List all available ONNX authorization models.

**Response:**
```json
{
  "models": [
    {
      "id": "simple_auth",
      "file": "simple_auth.onnx",
      "description": "Simple rule-based authorization",
      "inputCount": 5,
      "available": true
    }
  ]
}
```

**Model Schema:**
- `id` (string): Model identifier for API calls
- `file` (string): ONNX filename
- `description` (string): Human-readable description
- `inputCount` (number): Number of input features
- `available` (boolean): Whether model file exists

---

### POST /generate-proof

Generate a zero-knowledge proof for agent authorization.

**Request:**
```json
{
  "model": "simple_auth",
  "inputs": {
    "amount": "50",
    "balance": "1000",
    "velocity_1h": "20",
    "velocity_24h": "100",
    "vendor_trust": "80"
  }
}
```

**Request Schema:**
- `model` (string, required): Model ID from `/models` endpoint
- `inputs` (object, required): Authorization inputs (all strings representing integers)
  - `amount` (string): Transaction amount (scaled by 100, e.g., "50" = $0.50)
  - `balance` (string): Account balance (scaled by 100)
  - `velocity_1h` (string): 1-hour spending velocity
  - `velocity_24h` (string): 24-hour spending velocity
  - `vendor_trust` (string): Vendor trust score (0-100)

**Response (Success - arrives after 1-8 minutes):**
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
    "commitment": "0x1a2b3c...",
    "response": "0x4d5e6f...",
    "evaluation": "0x7g8h9i..."
  }
}
```

**Note**: Response takes 1-8 minutes due to comprehensive cryptographic verification.

**Response Schema:**
- `approved` (boolean): Whether transaction is authorized
- `output` (number): Raw model output value
- `verification` (boolean): Whether proof verified successfully
- `proofSize` (string): Estimated proof size
- `provingTime` (string): Time to generate proof (5-10 seconds)
- `verificationTime` (string): Time to verify proof cryptographically (40s - 7.5 minutes)
- `operations` (number): Number of ONNX operations in trace
- `zkmlProof` (object): Zero-knowledge proof components
  - `commitment` (string): Cryptographic commitment
  - `response` (string): Proof response
  - `evaluation` (string): Proof evaluation

**Response (Error):**
```json
{
  "error": "Proof generation failed",
  "message": "Model not found: invalid_model.onnx"
}
```

**Status Codes:**
- `200`: Proof generated successfully
- `400`: Invalid request (missing model or inputs)
- `500`: Proof generation error

**Timeout:**
- Recommended: 600 seconds (10 minutes)
- Total time: 1-8 minutes for proof generation + cryptographic verification
- Adjust with `timeout` query parameter: `/generate-proof?timeout=600000`

**Why it takes minutes**: The JOLT Atlas enhancements (Gather, Div, Cast, larger tensors) enable sophisticated authorization models. Comprehensive cryptographic verification of these enhanced capabilities requires thorough Spartan sumcheck validation (40s - 7.5 minutes).

---

### GET /validate-models

Validate that all configured models exist on disk.

**Response:**
```json
{
  "valid": true,
  "models": [
    {
      "id": "simple_auth",
      "file": "simple_auth.onnx",
      "exists": true,
      "path": "/path/to/simple_auth.onnx"
    }
  ]
}
```

---

## Integration Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function authorizeTransaction(amount, balance, velocity_1h, velocity_24h, vendor_trust) {
  try {
    // Production: https://zk-x402.com/api/generate-proof
    // Local: http://localhost:3001/api/generate-proof
    const response = await axios.post('https://zk-x402.com/api/generate-proof', {
      model: 'simple_threshold',
      inputs: {
        amount: amount.toString(),
        balance: balance.toString(),
        velocity_1h: velocity_1h.toString(),
        velocity_24h: velocity_24h.toString(),
        vendor_trust: vendor_trust.toString()
      }
    }, {
      timeout: 600000  // 10 minutes to accommodate cryptographic verification
    });

    if (response.data.approved && response.data.verification) {
      console.log('✅ Transaction authorized');
      console.log('Proof:', response.data.zkmlProof.commitment);
      return true;
    } else {
      console.log('❌ Transaction rejected');
      return false;
    }
  } catch (error) {
    console.error('Authorization failed:', error.message);
    return false;
  }
}

// Example usage
authorizeTransaction(50, 1000, 20, 100, 80);
```

### Python

```python
import requests

def authorize_transaction(amount, balance, velocity_1h, velocity_24h, vendor_trust):
    # Production: https://zk-x402.com/api/generate-proof
    # Local: http://localhost:3001/api/generate-proof
    response = requests.post('https://zk-x402.com/api/generate-proof',
        json={
            'model': 'simple_threshold',
            'inputs': {
                'amount': str(amount),
                'balance': str(balance),
                'velocity_1h': str(velocity_1h),
                'velocity_24h': str(velocity_24h),
                'vendor_trust': str(vendor_trust)
            }
        },
        timeout=600  # 10 minutes for cryptographic verification
    )

    if response.status_code == 200:
        result = response.json()
        if result['approved'] and result['verification']:
            print('✅ Transaction authorized')
            print(f"Proof size: {result['proofSize']}")
            return True
        else:
            print('❌ Transaction rejected')
            return False
    else:
        print(f'Error: {response.status_code}')
        return False

# Example usage
authorize_transaction(50, 1000, 20, 100, 80)
```

### cURL

```bash
#!/bin/bash

# Approved transaction (Production)
curl -X POST https://zk-x402.com/api/generate-proof \
  -H "Content-Type: application/json" \
  --max-time 600 \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": "50",
      "balance": "1000",
      "velocity_1h": "20",
      "velocity_24h": "100",
      "vendor_trust": "80"
    }
  }' | jq '.approved'

# Note: Response arrives in 1-6.5 minutes due to cryptographic verification

# Rejected transaction (excessive amount)
curl -X POST https://zk-x402.com/api/generate-proof \
  -H "Content-Type: application/json" \
  --max-time 600 \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": "200",
      "balance": "1000",
      "velocity_1h": "20",
      "velocity_24h": "100",
      "vendor_trust": "80"
    }
  }' | jq '.approved'

# For local development, use: http://localhost:3001/api/generate-proof
```

## Rate Limiting

**Current**: No rate limiting

**Recommendation for Production**:
```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many proof requests, please try again later.'
});

app.use('/api/generate-proof', limiter);
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not found` | ONNX file missing | Run `python3 create_demo_models.py` |
| `Invalid inputs` | Missing or malformed inputs | Check all required inputs are provided |
| `Proof generation failed` | Rust prover error | Check Rust build with `cargo build --release` |
| `Timeout` | Verification takes 1-8 minutes | Set timeout to 600 seconds (10 minutes) |

### Error Response Format

All errors return this structure:
```json
{
  "error": "Short error type",
  "message": "Detailed error message with context"
}
```

## Performance

### Response Times

| Model | Proof Generation | Verification | **Total Time** |
|-------|------------------|--------------|----------------|
| `simple_threshold` | 5-7s | 1-6 minutes | **1-6.5 minutes** |
| `velocity_1h` | 6-8s | 40s-4 minutes | **1-5 minutes** |
| `risk_neural` | 8-10s | 4-7.5 minutes | **5-8 minutes** |
| `multi_factor` | 6-8s | 5-7.5 minutes | **5-8 minutes** |
| `composite_scoring` | 8-10s | 5-7 minutes | **5-8 minutes** |

**Why verification takes minutes**: The JOLT Atlas enhancements (Gather, Div, Cast, larger tensors) enable sophisticated authorization models. Cryptographic verification requires comprehensive Spartan sumcheck validation.

### Integration Patterns

1. **Async Workflows**: Use webhooks for non-blocking proof generation
2. **Batch Processing**: Generate proofs overnight for scheduled transactions
3. **Pre-Computation**: Generate proofs in advance for predictable authorization needs
4. **Simulation First**: Use `/api/policies/:id/simulate` (<1ms) for testing before proof generation

## Security Considerations

### Input Validation

Always validate inputs before proof generation:

```javascript
function validateInputs(inputs) {
  const required = ['amount', 'balance', 'velocity_1h', 'velocity_24h', 'vendor_trust'];
  for (const field of required) {
    if (!inputs[field] || isNaN(parseInt(inputs[field]))) {
      throw new Error(`Invalid ${field}`);
    }
  }
}
```

### Proof Verification

**Critical**: Always verify that `verification: true` in the response before accepting authorization.

```javascript
if (response.data.approved && response.data.verification) {
  // OK to proceed
} else {
  // Reject transaction
}
```

### Production Deployment

- **HTTPS**: Always use HTTPS in production
- **Authentication**: Implement API key or JWT authentication
- **Rate Limiting**: Prevent abuse
- **Logging**: Log all authorization attempts
- **Monitoring**: Track proof generation failures

## Support

- **GitHub Issues**: https://github.com/yourusername/zkx402/issues
- **Documentation**: https://github.com/yourusername/zkx402/tree/main/docs
- **JOLT Atlas**: https://github.com/ICME-Lab/jolt-atlas

---

**Version**: 1.0.0
**Last Updated**: 2025-10-28
