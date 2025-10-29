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

**Response (Success):**
```json
{
  "approved": true,
  "output": 100,
  "verification": true,
  "proofSize": "15.2 KB",
  "verificationTime": "45ms",
  "operations": 21,
  "zkmlProof": {
    "commitment": "0x1a2b3c...",
    "response": "0x4d5e6f...",
    "evaluation": "0x7g8h9i..."
  }
}
```

**Response Schema:**
- `approved` (boolean): Whether transaction is authorized
- `output` (number): Raw model output value
- `verification` (boolean): Whether proof verified successfully
- `proofSize` (string): Estimated proof size
- `verificationTime` (string): Time to verify proof
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
- Default: 120 seconds
- Adjust with `timeout` query parameter: `/generate-proof?timeout=180000`

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
      model: 'simple_auth',
      inputs: {
        amount: amount.toString(),
        balance: balance.toString(),
        velocity_1h: velocity_1h.toString(),
        velocity_24h: velocity_24h.toString(),
        vendor_trust: vendor_trust.toString()
      }
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
    response = requests.post('https://zk-x402.com/api/generate-proof', json={
        'model': 'simple_auth',
        'inputs': {
            'amount': str(amount),
            'balance': str(balance),
            'velocity_1h': str(velocity_1h),
            'velocity_24h': str(velocity_24h),
            'vendor_trust': str(vendor_trust)
        }
    })

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
  -d '{
    "model": "simple_auth",
    "inputs": {
      "amount": "50",
      "balance": "1000",
      "velocity_1h": "20",
      "velocity_24h": "100",
      "vendor_trust": "80"
    }
  }' | jq '.approved'

# Rejected transaction (excessive amount)
curl -X POST https://zk-x402.com/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_auth",
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
| `Invalid inputs` | Missing or malformed inputs | Check all 5 inputs are provided |
| `Proof generation failed` | Rust prover error | Check Rust build with `cargo build --release` |
| `Timeout` | Proof took too long | Increase timeout or use simpler model |

### Error Response Format

All errors return this structure:
```json
{
  "error": "Short error type",
  "message": "Detailed error message with context"
}
```

## Performance

### Typical Response Times

| Model | Proof Generation | Verification | Total |
|-------|------------------|--------------|-------|
| `simple_auth` | 700ms | 45ms | ~750ms |
| `neural_auth` | 1500ms | 65ms | ~1.6s |
| `comparison_demo` | 300ms | 30ms | ~330ms |

### Optimization Tips

1. **Caching**: Cache proofs for identical inputs
2. **Batching**: Process multiple authorizations in parallel
3. **Model Selection**: Use simpler models when possible
4. **Hardware**: Proof generation is CPU-intensive, scale horizontally

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
