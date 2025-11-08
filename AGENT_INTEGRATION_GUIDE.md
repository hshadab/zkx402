# Agent Integration Guide for zkX402

## Overview

zkX402 is designed for **autonomous agent integration** with comprehensive discovery mechanisms, machine-readable schemas, and x402 protocol compliance.

This guide shows how AI agents can discover, integrate, and transact with zkX402.

---

## Quick Start for Agents

### 1. Discovery (Start Here)

**Primary Discovery Endpoint:**
```http
GET /.well-known/x402
```

Returns service capabilities, pricing, and available authorization models.

**Response:**
```json
{
  "service": "zkX402 Privacy-Preserving Authorization for AI Agents",
  "version": "1.3.0",
  "x402Version": 1,
  "capabilities": {
    "zkml_proofs": true,
    "max_model_params": 1024,
    "supported_onnx_ops": ["Gather", "Greater", "Less", ...]
  },
  "pricing": {
    "currency": "USDC",
    "network": "base",
    "chainId": 8453,
    "wallet": "0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91"
  },
  "endpoints": {
    "list_policies": "/api/policies",
    "generate_proof": "/api/generate-proof",
    "authorize": "/x402/authorize/:modelId"
  },
  "pre_built_policies": [...]
}
```

### 2. OpenAPI Specification

**Machine-Readable API Spec:**
```http
GET /openapi.yaml  # YAML format
GET /openapi.json  # JSON format
```

Full API specification with:
- All endpoints and parameters
- Request/response schemas
- Authentication methods
- Error codes and examples

---

## Integration Patterns

### Pattern 1: Direct Integration (No Payment)

**Use Case:** Testing, free tier (5 proofs/day)

```python
import requests

# 1. Discover available policies
response = requests.get('https://zkx402-agent-auth.onrender.com/api/policies')
policies = response.json()['policies']

# 2. Get schema for specific policy
policy_id = 'simple_threshold'
schema = requests.get(f'https://zkx402-agent-auth.onrender.com/api/policies/{policy_id}/schema').json()

# 3. Simulate (instant, no proof)
simulation = requests.post(
    f'https://zkx402-agent-auth.onrender.com/api/policies/{policy_id}/simulate',
    json={'inputs': {'amount': 100, 'balance': 500}}
).json()
print(f"Simulation: approved={simulation['approved']}")

# 4. Generate proof (1-8 minutes)
proof = requests.post(
    'https://zkx402-agent-auth.onrender.com/api/generate-proof',
    json={
        'model': policy_id,
        'inputs': {'amount': 100, 'balance': 500}
    }
).json()
print(f"Proof: approved={proof['approved']}, cached={proof.get('cached', False)}")
```

### Pattern 2: x402 Payment Flow (Production)

**Use Case:** Unlimited proofs, pay-per-use

```python
import requests
import base64
import json
from web3 import Web3

# 1. Request authorization (no payment yet)
response = requests.post(
    'https://zkx402-agent-auth.onrender.com/x402/authorize/simple_threshold',
    json={'inputs': {'amount': 100, 'balance': 500}}
)

# Expect 402 Payment Required
assert response.status_code == 402
payment_requirements = response.json()['accepts'][0]

# 2. Send USDC payment on Base
w3 = Web3(Web3.HTTPProvider('https://mainnet.base.org'))
usdc_address = payment_requirements['asset']['address']
pay_to = payment_requirements['payTo']
amount = int(payment_requirements['maxAmountRequired'])

# Send transaction (simplified - use your wallet)
tx_hash = send_usdc_payment(usdc_address, pay_to, amount)

# 3. Generate proof (off-chain)
proof_data = generate_local_proof(inputs)  # Your zkML proof

# 4. Create payment payload
payment_payload = {
    'x402Version': 1,
    'scheme': 'zkml-jolt',
    'network': 'base-mainnet',
    'payload': {
        'paymentTxHash': tx_hash,
        'modelId': 'simple_threshold',
        'zkmlProof': proof_data
    }
}

# 5. Encode payment header
payment_header = base64.b64encode(json.dumps(payment_payload).encode()).decode()

# 6. Submit with payment
final_response = requests.post(
    'https://zkx402-agent-auth.onrender.com/x402/authorize/simple_threshold',
    json={'inputs': {'amount': 100, 'balance': 500}},
    headers={'X-PAYMENT': payment_header}
)

# Success!
assert final_response.status_code == 200
result = final_response.json()
print(f"Authorized: {result['authorized']}")
```

### Pattern 3: Async with Webhooks

**Use Case:** Long-running operations, avoid polling

```python
# 1. Register webhook
webhook_response = requests.post(
    'https://zkx402-agent-auth.onrender.com/api/webhooks',
    json={
        'url': 'https://your-agent.com/webhooks/zkx402',
        'events': ['proof.completed', 'proof.failed']
    }
).json()

webhook_id = webhook_response['webhook_id']

# 2. Generate proof with webhook
proof_request = requests.post(
    'https://zkx402-agent-auth.onrender.com/api/generate-proof',
    json={
        'model': 'simple_threshold',
        'inputs': {'amount': 100, 'balance': 500},
        'webhook_id': webhook_id
    }
)

# 3. Your webhook receives notification when complete
# POST https://your-agent.com/webhooks/zkx402
# {
#   "event": "proof.completed",
#   "request_id": "req_xxx",
#   "proof": {...}
# }
```

---

## Discovery Mechanisms

### 1. x402 Well-Known Endpoint

**Standard:** [Coinbase x402 Protocol](https://github.com/coinbase/x402)

```
GET /.well-known/x402
```

**Use for:**
- Service capabilities and status
- Payment requirements and pricing
- Supported schemes/networks
- Endpoint discovery
- Available authorization models

**This is the primary discovery mechanism for service APIs using x402.**

### 2. OpenAPI Specification

**Standard:** [OpenAPI 3.0](https://swagger.io/specification/)

```
GET /openapi.yaml  # YAML format
GET /openapi.json  # JSON format
```

**Use for:**
- Programmatic API client generation
- Type definitions and schemas
- Complete endpoint documentation
- Request/response validation
- Testing and integration

### 3. Health Check

```
GET /health
```

**Returns:**
```json
{
  "status": "healthy",
  "service": "zkX402",
  "models": 14,
  "timestamp": "2025-11-08T..."
}
```

---

## Agent-Friendly Features

### 1. Free Tier for Testing

- **5 proofs per day** per IP
- No authentication required
- Full functionality
- Rate limit headers

### 2. Instant Simulation

- **<1ms response time**
- No proof generation
- Test logic before paying
- Endpoint: `/api/policies/{id}/simulate`

### 3. Caching for Performance

- **600-4800x faster** for repeated inputs
- Automatic SHA-256 hash-based caching
- 24-hour TTL
- Response includes `"cached": true`

### 4. Structured Errors

All errors include:
- Machine-readable error codes
- Human-readable messages
- Suggestions for resolution
- Request IDs for debugging

**Example:**
```json
{
  "error": "rate_limit_exceeded",
  "message": "Free tier limit of 5 proofs per day exceeded",
  "resetTime": "24 hours from first request",
  "upgradeOptions": {
    "x402Integration": {
      "endpoint": "/x402/authorize/:modelId",
      "documentation": "https://github.com/hshadab/zkx402"
    }
  }
}
```

### 5. Comprehensive Schemas

Every policy includes:
- JSON schema for inputs
- Type definitions
- Validation rules
- Example values

**Example Schema:**
```json
{
  "policyId": "simple_threshold",
  "inputs": [
    {
      "name": "amount",
      "type": "integer",
      "required": true,
      "description": "Transaction amount to authorize",
      "minimum": 0
    },
    {
      "name": "balance",
      "type": "integer",
      "required": true,
      "description": "Current account balance",
      "minimum": 0
    }
  ],
  "output": {
    "type": "boolean",
    "description": "true if approved, false if denied"
  }
}
```

---

## Payment Integration

### Base USDC Payments

**Network:** Base Mainnet (Chain ID: 8453)
**Token:** USDC (0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913)
**Decimals:** 6

**Payment Address:**
```
0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
```

### Pricing

| Model | Price (USDC) | Atomic Units | Time |
|-------|-------------|--------------|------|
| simple_threshold | $0.001 | 1000 | 1-6.5 min |
| percentage_limit | $0.0015 | 1500 | 1-6 min |
| velocity_1h | $0.002 | 2000 | 1-5 min |
| risk_neural | $0.06 | 60000 | 5-8 min |

### Payment Verification

```javascript
// zkX402 verifies payments on-chain
const verification = {
  txHash: '0xabc...',
  amount: '1000', // atomic units
  sender: '0x...',
  recipient: '0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91',
  timestamp: '2025-11-08T...',
  blockNumber: 12345678
}
```

---

## Error Handling

### Common Error Codes

| Code | Status | Description | Resolution |
|------|--------|-------------|------------|
| `model_not_found` | 404 | Policy/model doesn't exist | Check `/api/policies` for valid IDs |
| `missing_inputs` | 400 | Required inputs not provided | Check `/api/policies/{id}/schema` |
| `invalid_inputs` | 400 | Input validation failed | Verify types and ranges |
| `rate_limit_exceeded` | 429 | Free tier limit hit | Use x402 payment or wait 24h |
| `payment_required` | 402 | x402 payment needed | Follow payment flow |
| `payment_invalid` | 402 | Payment verification failed | Check tx hash and amount |
| `proof_generation_failed` | 500 | Internal proof error | Retry or contact support |

### Retry Logic

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def generate_proof_with_retry(model_id, inputs):
    response = requests.post(
        'https://zkx402-agent-auth.onrender.com/api/generate-proof',
        json={'model': model_id, 'inputs': inputs},
        timeout=600  # 10 minutes for proof generation
    )
    response.raise_for_status()
    return response.json()
```

---

## Performance Optimization

### 1. Use Simulation First

```python
# Fast check (< 1ms)
simulation = simulate_policy(model_id, inputs)
if not simulation['approved']:
    return "Denied"  # Don't waste time/money on proof

# Only generate proof if needed
proof = generate_proof(model_id, inputs)  # 1-8 minutes
```

### 2. Leverage Caching

```python
# Same inputs = instant response from cache
proof1 = generate_proof('simple_threshold', {'amount': 100, 'balance': 500})
# ... 5 minutes later

proof2 = generate_proof('simple_threshold', {'amount': 100, 'balance': 500})
# Returns in <100ms with cached=true
```

### 3. Batch Operations

```python
# Generate multiple proofs in parallel
from concurrent.futures import ThreadPoolExecutor

models = ['simple_threshold', 'velocity_1h', 'vendor_trust']
inputs_list = [...]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(generate_proof, model, inputs)
        for model, inputs in zip(models, inputs_list)
    ]
    results = [f.result() for f in futures]
```

---

## Security Best Practices

### 1. Verify Proofs

```python
# Always verify the proof was issued by zkX402
def verify_proof(proof_result):
    # Check signature
    assert proof_result['verification']['verified'] == True
    # Check timestamp
    assert is_recent(proof_result['timestamp'])
    # Check model matches
    assert proof_result['modelId'] == expected_model
    return proof_result['approved']
```

### 2. Validate Inputs

```python
# Get schema first
schema = get_policy_schema(model_id)

# Validate before submitting
def validate_inputs(inputs, schema):
    for field in schema['inputs']:
        if field['required'] and field['name'] not in inputs:
            raise ValueError(f"Missing required field: {field['name']}")
        if 'minimum' in field and inputs[field['name']] < field['minimum']:
            raise ValueError(f"{field['name']} below minimum")
    return True
```

### 3. Handle Rate Limits

```python
def smart_proof_generation(model_id, inputs):
    try:
        return generate_proof(model_id, inputs)
    except RateLimitError as e:
        # Switch to paid tier
        return generate_proof_with_payment(model_id, inputs)
```

---

## SDK Integration

### Python SDK

```python
from zkx402 import ZKX402Client

# Initialize client
client = ZKX402Client(
    base_url='https://zkx402-agent-auth.onrender.com',
    wallet_address='0x...',  # For x402 payments
    private_key='0x...'  # For signing
)

# Auto-discover capabilities
capabilities = client.discover()

# Generate proof (handles caching, retries, errors)
proof = client.generate_proof(
    model='simple_threshold',
    inputs={'amount': 100, 'balance': 500},
    payment_method='x402'  # or 'free_tier'
)

# Verify proof
assert client.verify_proof(proof)
```

### JavaScript SDK (Coming Soon)

```javascript
import { ZKX402 } from '@zkx402/sdk';

const client = new ZKX402({
  baseUrl: 'https://zkx402-agent-auth.onrender.com'
});

const proof = await client.generateProof({
  model: 'simple_threshold',
  inputs: { amount: 100, balance: 500 }
});
```

---

## Testing

### Sandbox Environment

Use `http://localhost:3001` for testing without hitting production limits.

### Test Models

Four test models available for integration testing:

- `test_less`: Tests Less operation
- `test_identity`: Tests Identity operation
- `test_clip`: Tests Clip operation
- `test_slice`: Tests Slice operation

**All free, instant, for testing only.**

### Example Test Suite

```python
import pytest
from zkx402 import ZKX402Client

@pytest.fixture
def client():
    return ZKX402Client(base_url='http://localhost:3001')

def test_discovery(client):
    discovery = client.discover()
    assert discovery['status'] == 'production'
    assert len(discovery['pre_built_policies']) >= 10

def test_simulation(client):
    result = client.simulate('simple_threshold', {'amount': 100, 'balance': 500})
    assert result['approved'] == True

def test_proof_generation(client):
    proof = client.generate_proof('test_less', {'a': 5, 'b': 10})
    assert proof['approved'] == True
    assert 'zkmlProof' in proof

def test_caching(client):
    # First call - cache miss
    proof1 = client.generate_proof('test_less', {'a': 5, 'b': 10})
    assert proof1.get('cached') == False

    # Second call - cache hit
    proof2 = client.generate_proof('test_less', {'a': 5, 'b': 10})
    assert proof2['cached'] == True
```

---

## Monitoring & Analytics

### Request Tracking

Every request includes a `request_id` for debugging:

```json
{
  "proof": {...},
  "request_id": "req_1699454625123_abc123"
}
```

Use this when reporting issues.

### Cache Statistics

```http
GET /api/cache/stats
```

**Response:**
```json
{
  "hits": 142,
  "misses": 58,
  "errors": 0,
  "hitRate": "71.00%",
  "enabled": true,
  "connected": true
}
```

---

## Support

### Documentation
- **GitHub:** https://github.com/hshadab/zkx402
- **API Reference:** /openapi.yaml
- **Quick Start:** /QUICKSTART.md

### Issues
- **GitHub Issues:** https://github.com/hshadab/zkx402/issues
- **Include:** request_id, error message, timestamps

### Community
- **Discussions:** https://github.com/hshadab/zkx402/discussions
- **Examples:** https://github.com/hshadab/zkx402/tree/main/examples

---

## Compliance

### Standards Supported

- ✅ **x402 Protocol** (Coinbase)
- ✅ **OpenAPI 3.0**
- ✅ **ERC-20** (USDC payments)
- ✅ **ONNX** (model format)

### Privacy

- Zero-knowledge proofs preserve input privacy
- Proof results cached 24 hours
- Logs retained 30 days
- GDPR compliant

### Security

- TLS 1.3 encryption
- ECDSA signing
- On-chain payment verification
- Audit logs

---

## Next Steps

1. **Try the free tier:** 5 proofs/day, no setup
2. **Read OpenAPI spec:** Understand all endpoints
3. **Test simulation:** Instant results, no cost
4. **Integrate x402:** Unlimited production access
5. **Monitor performance:** Use cache stats

**Start integrating now:** `GET /.well-known/x402`
