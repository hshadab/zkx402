# Agent Integration Guide

**For Autonomous x402 Agents**

This guide explains how autonomous agents can discover, integrate, and use zkX402's privacy-preserving authorization policies programmatically.

## Quick Start for Agents

### 1. Service Discovery

Start by fetching the x402 service descriptor:

```bash
curl https://your-server.com/.well-known/x402
```

**Response includes:**
- `capabilities`: Technical limits (max_model_params, supported operations)
- `pricing`: USDC payment details on Base L2
- `endpoints`: All available API URLs
- `pre_built_policies`: Array of 14 ready-to-use authorization policies

### 2. Discover Available Policies

```bash
curl https://your-server.com/api/policies
```

**Returns:** Machine-readable catalog of all 14 authorization policies with:
- Policy metadata (ID, name, description, category)
- Input schema with types and descriptions
- Example requests (approve/deny scenarios)
- Pricing and performance metrics
- Direct endpoint URLs

### 3. Get Policy Schema

```bash
curl https://your-server.com/api/policies/simple_threshold/schema
```

**Returns:** Detailed schema for a specific policy including:
- Input/output specifications
- Validation rules
- Usage examples
- Pricing information

### 4. Simulate Policy (Fast Testing)

```bash
curl -X POST https://your-server.com/api/policies/simple_threshold/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "amount": 5000,
      "balance": 10000
    }
  }'
```

**Returns:**
```json
{
  "simulation": true,
  "approved": true,
  "output": 1,
  "policy_id": "simple_threshold",
  "policy_name": "Simple Threshold",
  "inputs": {
    "amount": 5000,
    "balance": 10000
  },
  "execution_time_ms": "<1ms (simulation only)",
  "note": "This is a simulation without zkML proof. Use /api/generate-proof to get a verifiable proof.",
  "proof_generation": {
    "endpoint": "https://your-server.com/api/generate-proof",
    "estimated_time": "~0.5s",
    "estimated_cost": "0.0001"
  }
}
```

**Benefits:**
- Instant results (< 1ms) without generating expensive zkML proofs
- Test policies before committing to proof generation
- No payment required for simulation
- Perfect for testing and validation workflows

### 5. Generate Proof

```bash
curl -X POST https://your-server.com/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": "5000",
      "balance": "10000"
    }
  }'
```

**Returns:**
```json
{
  "approved": true,
  "output": 1,
  "proof": "0x...",
  "verification": {
    "verified": true
  },
  "proving_time": "732ms",
  "model_type": "simple_threshold"
}
```

### 6. Async Proof Generation with Webhooks

For long-running proof generation, agents can register webhooks to receive async notifications when proofs complete.

#### Step 1: Register a Webhook

```bash
curl -X POST https://your-server.com/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "callback_url": "https://your-agent.com/webhook/proof-complete",
    "metadata": {
      "agent_id": "agent_123",
      "task_id": "task_456"
    }
  }'
```

**Response:**
```json
{
  "webhook_id": "wh_1234567890_abc123",
  "callback_url": "https://your-agent.com/webhook/proof-complete",
  "metadata": {
    "agent_id": "agent_123",
    "task_id": "task_456"
  },
  "created_at": "2025-10-29T12:00:00.000Z",
  "status": "active"
}
```

#### Step 2: Generate Proof with Webhook

```bash
curl -X POST https://your-server.com/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": "5000",
      "balance": "10000"
    },
    "webhook_id": "wh_1234567890_abc123"
  }'
```

The proof is generated synchronously, but upon completion, the webhook is triggered automatically.

#### Step 3: Receive Webhook Notification

Your webhook endpoint will receive a POST request with:

```json
{
  "event": "proof.completed",
  "webhook_id": "wh_1234567890_abc123",
  "timestamp": "2025-10-29T12:00:05.000Z",
  "data": {
    "request_id": "req_1234567890_xyz789",
    "policy_id": "simple_threshold",
    "inputs": {
      "amount": "5000",
      "balance": "10000"
    },
    "approved": true,
    "output": 1,
    "verification": {
      "verified": true
    },
    "proofTime": 732,
    "modelName": "Simple Threshold Check"
  }
}
```

**Webhook Headers:**
- `Content-Type: application/json`
- `X-zkX402-Event: proof.completed`
- `X-zkX402-Webhook-ID: wh_1234567890_abc123`

#### Managing Webhooks

**Get webhook details:**
```bash
GET /api/webhooks/{webhook_id}
```

**List all webhooks:**
```bash
GET /api/webhooks
```

**Delete webhook:**
```bash
DELETE /api/webhooks/{webhook_id}
```

#### Benefits of Webhooks

- **Non-blocking**: Your agent doesn't need to wait for proof generation
- **Reliable**: Webhook delivery is tracked and logged
- **Flexible**: Attach custom metadata to track context
- **Scalable**: Generate multiple proofs in parallel without polling

## Available Authorization Policies

### Policy Tiers

zkX402 provides **14 total policies**, organized into tiers:

**Featured Policies (4)** - Advanced zkML showcasing true capabilities:
- `percentage_limit` - Arithmetic computation (multiplication + division)
- `multi_factor` - Multi-check composition (30 operations)
- `composite_scoring` - Weighted scoring model (72 operations)
- `risk_neural` - Real neural network (47 operations)

**Simple Policies (6)** - Basic comparisons (also available via API):
- `simple_threshold`, `vendor_trust`, `velocity_1h`, `velocity_24h`, `daily_limit`, `age_gate`

**Test Policies (4)** - Internal testing only:
- `test_less`, `test_identity`, `test_clip`, `test_slice`

The UI displays only the 4 featured models to highlight advanced zkML capabilities. All 14 policies remain available via API for programmatic access.

### Financial Authorization

**simple_threshold** - Basic balance check
- **Inputs**: `amount`, `balance`
- **Logic**: Approve if amount < balance
- **Use case**: Simple spending authorization
- **Performance**: ~0.5s proof time

**percentage_limit** - Percentage-based spending cap
- **Inputs**: `amount`, `balance`, `max_percentage`
- **Logic**: Approve if amount < (balance * max_percentage / 100)
- **Use case**: Limit spending to % of available funds
- **Performance**: ~1.1s proof time

**velocity_1h** - Hourly spending velocity
- **Inputs**: `amount`, `spent_1h`, `limit_1h`
- **Logic**: Approve if (spent_1h + amount) < limit_1h
- **Use case**: Rate limiting for high-frequency trading
- **Performance**: ~1.2s proof time

**velocity_24h** - Daily spending velocity
- **Inputs**: `amount`, `spent_24h`, `limit_24h`
- **Logic**: Approve if (spent_24h + amount) < limit_24h
- **Use case**: Daily spending caps
- **Performance**: ~1.2s proof time

**daily_limit** - Daily spending cap
- **Inputs**: `amount`, `daily_spent`, `daily_cap`
- **Logic**: Approve if (daily_spent + amount) < daily_cap
- **Use case**: Budget enforcement
- **Performance**: ~1.2s proof time

### Trust & Reputation

**vendor_trust** - Vendor reputation check
- **Inputs**: `vendor_trust`, `min_trust`
- **Logic**: Approve if vendor_trust >= min_trust
- **Use case**: Whitelist high-trust vendors
- **Performance**: ~0.6s proof time

### Access Control

**age_gate** - Age verification
- **Inputs**: `age`, `min_age`
- **Logic**: Approve if age >= min_age
- **Use case**: Age-restricted services
- **Performance**: ~0.6s proof time

### Multi-Factor Policies

**multi_factor** - Combined financial + trust checks
- **Inputs**: `amount`, `balance`, `spent_24h`, `limit_24h`, `vendor_trust`, `min_trust`
- **Logic**: Balance sufficient AND velocity within limits AND vendor trusted
- **Use case**: Comprehensive transaction authorization
- **Performance**: ~4.1s proof time

**composite_scoring** - Weighted scoring model
- **Inputs**: `amount`, `balance`, `vendor_trust`, `user_history`
- **Logic**: Weighted combination of multiple factors
- **Use case**: Risk-based authorization
- **Performance**: ~2.8s proof time

### Neural Network Policies

**risk_neural** - ML-based risk scoring
- **Inputs**: `amount`, `balance`, `velocity_1h`, `velocity_24h`, `vendor_trust`
- **Logic**: Trained neural network (8→4 hidden layers)
- **Use case**: Complex pattern recognition for fraud detection
- **Performance**: ~8.7s proof time

## Integration Patterns

### Pattern 1: Discovery-Driven Integration

```python
import requests

# 1. Discover service
discovery = requests.get("https://server.com/.well-known/x402").json()
print(f"Service: {discovery['service']}")
print(f"Available policies: {discovery['total_policies']}")

# 2. List policies
policies = requests.get(discovery['endpoints']['list_policies']).json()

# 3. Find suitable policy
policy = next(p for p in policies['policies'] if p['category'] == 'financial')
print(f"Using policy: {policy['name']}")

# 4. Get detailed schema
schema = requests.get(policy['schema_url']).json()

# 5. Generate proof
proof_response = requests.post(
    discovery['endpoints']['generate_proof'],
    json={
        "model": policy['id'],
        "inputs": {
            "amount": "5000",
            "balance": "10000"
        }
    }
).json()

if proof_response['approved']:
    print("✓ Authorization approved")
    print(f"Proof: {proof_response['proof'][:64]}...")
```

### Pattern 2: Policy Selection by Use Case

```javascript
// Fetch policy catalog
const { policies } = await fetch('/api/policies').then(r => r.json());

// Filter by category
const financialPolicies = policies.filter(p => p.category === 'financial');

// Select by complexity
const simplePolicy = financialPolicies.find(p => p.complexity === 'simple');

// Or by performance
const fastPolicy = policies.reduce((fastest, p) =>
  p.avg_proof_time_ms < fastest.avg_proof_time_ms ? p : fastest
);

// Generate proof
const proof = await fetch('/api/generate-proof', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: simplePolicy.id,
    inputs: simplePolicy.example.approve
  })
}).then(r => r.json());
```

### Pattern 3: Multi-Policy Evaluation

```python
# Evaluate multiple policies in parallel for different risk levels
import asyncio
import aiohttp

async def evaluate_policies(transaction):
    async with aiohttp.ClientSession() as session:
        # Get policies
        async with session.get('/api/policies') as resp:
            policies_data = await resp.json()

        # Evaluate relevant policies
        tasks = []
        for policy in policies_data['policies']:
            if policy['category'] in ['financial', 'multi-factor']:
                task = generate_proof(session, policy['id'], transaction)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Aggregate results
        approved_count = sum(1 for r in results if r['approved'])
        return {
            'total_policies': len(results),
            'approved': approved_count,
            'consensus': approved_count / len(results)
        }

async def generate_proof(session, model_id, inputs):
    async with session.post('/api/generate-proof', json={
        'model': model_id,
        'inputs': inputs
    }) as resp:
        return await resp.json()
```

## Technical Constraints

### JOLT Atlas Limitations

**Maximum Model Size:**
- Max tensor size: 1,024 elements
- Max operations: ~100 per model
- Integer-only operations (scaled fixed-point for decimals)

**Supported Input Types:**
- `int8`, `int16`, `int32`
- `float32` (converted to scaled integers internally)

**Proof Generation Times:**
- Simple policies (2-10 ops): 0.5-1.5s
- Medium policies (11-30 ops): 1.5-4s
- Complex policies (31-100 ops): 4-9s

### Payment Requirements

**Network:** Base L2
**Currency:** USDC
**Pricing:** Per-proof, varies by model complexity
- Simple: $0.0001-0.0005 USDC
- Medium: $0.0005-0.002 USDC
- Complex: $0.002-0.01 USDC

## Error Handling

### Common Errors

**404 - Policy Not Found**
```json
{
  "error": "Policy not found"
}
```
**Solution:** Check available policies via `/api/policies`

**400 - Invalid Input**
```json
{
  "error": "Missing required input: balance"
}
```
**Solution:** Validate inputs against policy schema

**402 - Payment Required**
```json
{
  "error": "Insufficient payment",
  "required": "0.001 USDC",
  "provided": "0"
}
```
**Solution:** Include payment with proof request

**500 - Proof Generation Failed**
```json
{
  "error": "Proof generation failed",
  "details": "Tensor size exceeds limit"
}
```
**Solution:** Use simpler policy or reduce input complexity

## Best Practices

### 1. Cache Policy Metadata
- Fetch `.well-known/x402` once on startup
- Cache policy schemas for frequently used policies
- Refresh every 24 hours or on version change

### 2. Select Appropriate Policies
- Use `simple_threshold` for basic checks (fastest)
- Use `multi_factor` for comprehensive authorization
- Use `risk_neural` only when ML insights are needed (slowest)

### 3. Handle Proofs Securely
- Store proofs for audit trails
- Verify proofs before accepting authorization
- Include proof metadata (timestamp, policy ID) in logs

### 4. Monitor Performance
- Track proof generation times
- Set timeouts (recommend 15s for complex policies)
- Fall back to simpler policies on timeout

### 5. Optimize for Cost
- Batch similar transactions when possible
- Use caching for repeated authorization checks
- Choose least complex policy that meets requirements

## Example: Complete Agent Flow

```python
class zkX402Agent:
    def __init__(self, server_url):
        self.server = server_url
        self.discovery = None
        self.policies = None

    async def initialize(self):
        """Discover service capabilities"""
        self.discovery = await self.fetch('/.well-known/x402')
        self.policies = await self.fetch('/api/policies')
        print(f"Initialized with {self.policies['total_policies']} policies")

    async def authorize_transaction(self, amount, balance):
        """Authorize a transaction with zero-knowledge proof"""
        # Select policy
        policy = self.select_policy('simple_threshold')

        # Generate proof
        proof = await self.generate_proof(
            policy['id'],
            {'amount': amount, 'balance': balance}
        )

        if proof['approved']:
            # Transaction authorized
            return {
                'authorized': True,
                'proof': proof['proof'],
                'policy': policy['name']
            }
        else:
            return {'authorized': False, 'reason': 'Policy denied'}

    def select_policy(self, policy_id):
        """Select policy by ID"""
        return next(p for p in self.policies['policies'] if p['id'] == policy_id)

    async def generate_proof(self, model_id, inputs):
        """Generate zero-knowledge proof"""
        return await self.post('/api/generate-proof', {
            'model': model_id,
            'inputs': inputs
        })

    async def fetch(self, path):
        # HTTP GET implementation
        pass

    async def post(self, path, data):
        # HTTP POST implementation
        pass
```

## Support & Resources

- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)
- **Payment Guide**: See [PAYMENT_GUIDE.md](PAYMENT_GUIDE.md)
- **Model Catalog**: Browse 14 pre-built policies at `/api/policies`
- **GitHub**: https://github.com/hshadab/zkx402
- **Issues**: Report bugs or request features on GitHub

## Next Steps

1. **Test Integration**: Start with `/api/policies` to explore available policies
2. **Run Examples**: Try the example code snippets above
3. **Deploy**: Integrate into your autonomous agent workflow
4. **Monitor**: Track proof generation performance and costs
5. **Optimize**: Select appropriate policies for your use case
