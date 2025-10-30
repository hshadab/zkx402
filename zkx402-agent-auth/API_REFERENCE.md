# zkX402 API Reference

**Version:** 1.3.0
**Base URL:** `https://your-server.com`

Complete API reference for zkX402 privacy-preserving authorization service.

## Table of Contents

1. [Discovery Endpoints](#discovery-endpoints)
2. [Policy Endpoints](#policy-endpoints)
3. [Proof Generation](#proof-generation)
4. [Webhook Endpoints](#webhook-endpoints)
5. [Data Models](#data-models)
6. [Error Responses](#error-responses)

---

## Discovery Endpoints

### GET `/.well-known/x402`

Service discovery endpoint for x402 protocol compliance.

**Description:** Returns service metadata, capabilities, pricing, and available policies.

**Request:**
```bash
GET /.well-known/x402
```

**Response:** `200 OK`
```json
{
  "service": "zkX402 Privacy-Preserving Authorization for AI Agents",
  "version": "1.3.0",
  "status": "production",
  "lastUpdated": "2025-10-29",
  "description": "Privacy-preserving authorization using JOLT Atlas zkML proofs with Base USDC payments...",
  "x402Version": 1,

  "capabilities": {
    "zkml_proofs": true,
    "max_model_params": 1024,
    "max_operations": 100,
    "supported_input_types": ["int8", "int16", "int32", "float32"],
    "proof_time_range": "0.5s - 9s",
    "supported_onnx_ops": [
      "Gather", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
      "Div", "Cast", "Slice", "Identity", "Add", "Sub", "Mul",
      "Clip", "MatMul"
    ],
    "custom_model_upload": false,
    "policy_composition": false
  },

  "pricing": {
    "currency": "USDC",
    "network": "base",
    "chainId": 8453,
    "wallet": "0x...",
    "explorer": "https://basescan.org",
    "tokenAddress": "0x..."
  },

  "endpoints": {
    "list_policies": "http://localhost:3001/api/policies",
    "get_policy_schema": "http://localhost:3001/api/policies/:id/schema",
    "generate_proof": "http://localhost:3001/api/generate-proof",
    "verify_proof": "http://localhost:3001/x402/verify-proof",
    "authorize": "http://localhost:3001/x402/authorize/:modelId",
    "discovery": "http://localhost:3001/.well-known/x402",
    "health": "http://localhost:3001/health"
  },

  "pre_built_policies": [
    {
      "id": "simple_threshold",
      "name": "Simple Threshold Check",
      "description": "Approve if amount < balance",
      "category": "financial",
      "use_case": "Basic spending authorization without revealing balance",
      "price_usdc": "0.0001",
      "price_atomic": "100",
      "avg_proof_time": "~0.5s",
      "operations": 2,
      "complexity": "simple",
      "inputs": ["amount", "balance"],
      "endpoint": "http://localhost:3001/x402/authorize/simple_threshold",
      "schema_url": "http://localhost:3001/api/policies/simple_threshold/schema"
    }
    // ... 13 more policies
  ],

  "documentation": "https://github.com/hshadab/zkx402",
  "sdk": {
    "python": "https://github.com/hshadab/zkx402-sdk-python",
    "javascript": "https://github.com/hshadab/zkx402-sdk-js"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service name |
| `version` | string | API version (semver) |
| `status` | string | Service status: `production`, `staging`, `maintenance` |
| `capabilities` | object | Technical capabilities and limitations |
| `pricing` | object | Payment network and currency details |
| `endpoints` | object | All available API endpoints |
| `pre_built_policies` | array | List of ready-to-use authorization policies |

---

## Policy Endpoints

### Policy Tiers: Featured vs All

zkX402 provides 14 total authorization policies organized into tiers:

**Featured Policies (4)** - Advanced zkML models showcased in the UI:
- `percentage_limit` - Actual arithmetic computation (multiplication + division)
- `multi_factor` - Multi-check composition (30 operations)
- `composite_scoring` - Weighted scoring model (72 operations)
- `risk_neural` - Real neural network for risk scoring (47 operations)

**Production Policies (10)** - All policies including simpler ones:
- Featured models (above) plus basic comparison models like `simple_threshold`, `vendor_trust`, `velocity_1h`, etc.
- All 14 policies available via API for programmatic access
- UI displays only the 4 featured models to highlight advanced zkML capabilities

**Test Policies (4)** - Internal testing only:
- `test_less`, `test_identity`, `test_clip`, `test_slice`
- Available in API but not shown in UI

### GET `/api/policies`

List all available authorization policies with machine-readable metadata.

**Note:** This endpoint returns all 14 policies. The UI displays only the 4 featured models to showcase advanced zkML capabilities, but agents can use any of the 14 policies programmatically.

**Request:**
```bash
GET /api/policies
```

**Response:** `200 OK`
```json
{
  "version": "1.3.0",
  "total_policies": 14,
  "policies": [
    {
      "id": "simple_threshold",
      "name": "Simple Threshold Check",
      "description": "Approve if amount < balance",
      "category": "financial",
      "complexity": "simple",
      "operations": 2,
      "avg_proof_time_ms": 500,
      "price_usdc": "0.0001",
      "price_atomic": "100",

      "schema": {
        "inputs": {
          "amount": {
            "type": "int32",
            "description": "Transaction amount to authorize",
            "required": true
          },
          "balance": {
            "type": "int32",
            "description": "Current account balance",
            "required": true
          }
        },
        "output": {
          "type": "int32",
          "description": "1 = approved, 0 = denied"
        }
      },

      "example": {
        "approve": {
          "amount": 5000,
          "balance": 10000
        },
        "deny": {
          "amount": 15000,
          "balance": 10000
        }
      },

      "use_case": "Basic spending authorization without revealing balance",
      "endpoint": "http://localhost:3001/x402/authorize/simple_threshold",
      "schema_url": "http://localhost:3001/api/policies/simple_threshold/schema"
    }
    // ... 13 more policies
  ],

  "capabilities": {
    "max_model_params": 1024,
    "max_operations": 100,
    "supported_types": ["int8", "int16", "int32", "float32"]
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | API version |
| `total_policies` | number | Total number of policies |
| `policies` | array | Array of policy objects |
| `policies[].id` | string | Unique policy identifier |
| `policies[].complexity` | string | `simple`, `medium`, or `advanced` |
| `policies[].avg_proof_time_ms` | number | Average proof generation time in milliseconds |
| `policies[].schema` | object | Input/output schema |
| `policies[].example` | object | Example inputs for approve/deny scenarios |

**Categories:**

- `financial` - Balance checks, spending limits
- `trust` - Vendor reputation, trust scores
- `access` - Age gates, permissions
- `multi-factor` - Combined authorization rules
- `neural` - ML-based risk scoring

**Complexity Levels:**

- `simple`: 2-10 operations, ~0.5-1.5s proof time
- `medium`: 11-30 operations, ~1.5-4s proof time
- `advanced`: 31-100 operations, ~4-9s proof time

---

### GET `/api/policies/:id/schema`

Get detailed schema for a specific policy.

**Request:**
```bash
GET /api/policies/simple_threshold/schema
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Policy ID (e.g., `simple_threshold`) |

**Response:** `200 OK`
```json
{
  "policy_id": "simple_threshold",
  "name": "Simple Threshold Check",
  "description": "Approve if amount < balance",
  "category": "financial",
  "version": "1.0.0",

  "schema": {
    "inputs": {
      "amount": {
        "type": "int32",
        "description": "Transaction amount to authorize",
        "required": true,
        "validation": {
          "type": "integer",
          "minimum": 0
        }
      },
      "balance": {
        "type": "int32",
        "description": "Current account balance",
        "required": true,
        "validation": {
          "type": "integer",
          "minimum": 0
        }
      }
    },

    "output": {
      "type": "int32",
      "description": "1 = approved (authorization granted), 0 = denied (authorization rejected)",
      "values": {
        "1": "approved",
        "0": "denied"
      }
    }
  },

  "examples": {
    "approve": {
      "description": "Example that will be approved",
      "inputs": {
        "amount": 5000,
        "balance": 10000
      },
      "expected_output": 1
    },
    "deny": {
      "description": "Example that will be denied",
      "inputs": {
        "amount": 15000,
        "balance": 10000
      },
      "expected_output": 0
    }
  },

  "pricing": {
    "price_usdc": "0.0001",
    "price_atomic": "100",
    "currency": "USDC",
    "network": "base"
  },

  "performance": {
    "operations": 2,
    "avg_proof_time": "~0.5s",
    "complexity": "simple"
  },

  "usage": {
    "endpoint": "http://localhost:3001/api/generate-proof",
    "method": "POST",
    "request_format": {
      "model": "simple_threshold",
      "inputs": {
        "amount": "<value>",
        "balance": "<value>"
      }
    }
  }
}
```

**Error Responses:**

- `404 Not Found` - Policy ID does not exist

---

### POST `/api/policies/:id/simulate`

Simulate policy execution without generating zkML proof. Fast and free for testing.

**Request:**
```bash
POST /api/policies/simple_threshold/simulate
Content-Type: application/json

{
  "inputs": {
    "amount": 5000,
    "balance": 10000
  }
}
```

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Policy ID (e.g., `simple_threshold`) |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inputs` | object | Yes | Input values matching policy schema |

**Response:** `200 OK`
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
    "endpoint": "http://localhost:3001/api/generate-proof",
    "estimated_time": "~0.5s",
    "estimated_cost": "0.0001"
  }
}
```

**Benefits:**
- Instant results (< 1ms) without zkML proof generation
- No payment required
- Perfect for testing policies before committing to proof generation
- Returns estimated cost and time for actual proof generation

**Error Responses:**

- `400 Bad Request` - Missing or invalid inputs
- `404 Not Found` - Policy ID does not exist
- `500 Internal Server Error` - Simulation failed

---

## Proof Generation

### POST `/api/generate-proof`

Generate zero-knowledge proof for authorization policy.

**Request:**
```bash
POST /api/generate-proof
Content-Type: application/json

{
  "model": "simple_threshold",
  "inputs": {
    "amount": "5000",
    "balance": "10000"
  }
}
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Policy ID to evaluate |
| `inputs` | object | Yes | Input values matching policy schema |

**Response:** `200 OK` (Approved)
```json
{
  "approved": true,
  "output": 1,
  "proof": "0x1a2b3c4d...",
  "verification": {
    "verified": true,
    "verifier": "jolt_atlas_v1",
    "timestamp": "2025-10-29T12:34:56.789Z"
  },
  "proving_time": "732ms",
  "proof_size": "15360",
  "model_type": "simple_threshold",
  "inputs_hash": "0xabc123...",
  "payment": {
    "required": "100",
    "currency": "USDC",
    "network": "base"
  }
}
```

**Response:** `200 OK` (Denied)
```json
{
  "approved": false,
  "output": 0,
  "proof": "0x1a2b3c4d...",
  "verification": {
    "verified": true,
    "verifier": "jolt_atlas_v1",
    "timestamp": "2025-10-29T12:34:56.789Z"
  },
  "proving_time": "689ms",
  "proof_size": "15360",
  "model_type": "simple_threshold",
  "inputs_hash": "0xabc123...",
  "payment": {
    "required": "100",
    "currency": "USDC",
    "network": "base"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `approved` | boolean | Whether authorization was granted |
| `output` | number | Policy output (1=approved, 0=denied) |
| `proof` | string | Zero-knowledge proof (hex-encoded) |
| `verification` | object | Proof verification details |
| `proving_time` | string | Time taken to generate proof |
| `proof_size` | string | Proof size in bytes |
| `model_type` | string | Policy ID that was evaluated |
| `inputs_hash` | string | Hash of inputs (for audit) |

**Error Responses:**

- `400 Bad Request` - Missing or invalid inputs
- `402 Payment Required` - Insufficient payment
- `404 Not Found` - Model ID does not exist
- `500 Internal Server Error` - Proof generation failed

---

## Webhook Endpoints

Webhooks allow agents to receive async notifications when proofs complete, enabling non-blocking workflows for long-running proof generation.

### POST `/api/webhooks`

Register a new webhook for proof completion notifications.

**Request:**
```bash
POST /api/webhooks
Content-Type: application/json
```

```json
{
  "callback_url": "https://your-agent.com/webhook/proof-complete",
  "metadata": {
    "agent_id": "agent_123",
    "task_id": "task_456"
  }
}
```

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `callback_url` | string | ✓ | Your webhook endpoint URL |
| `metadata` | object | | Optional metadata to track webhook context |

**Response:** `201 Created`
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

**Error Responses:**
- `400 Bad Request` - Missing callback_url

---

### GET `/api/webhooks/:id`

Get details about a specific webhook.

**Request:**
```bash
GET /api/webhooks/wh_1234567890_abc123
```

**Response:** `200 OK`
```json
{
  "webhook_id": "wh_1234567890_abc123",
  "callback_url": "https://your-agent.com/webhook/proof-complete",
  "metadata": {
    "agent_id": "agent_123",
    "task_id": "task_456"
  },
  "created_at": "2025-10-29T12:00:00.000Z",
  "deliveries": 5,
  "recent_deliveries": [
    {
      "id": "delivery_1234567890",
      "timestamp": "2025-10-29T12:00:05.000Z",
      "status": "delivered",
      "response_status": 200
    }
  ]
}
```

**Error Responses:**
- `404 Not Found` - Webhook not found

---

### DELETE `/api/webhooks/:id`

Delete a webhook.

**Request:**
```bash
DELETE /api/webhooks/wh_1234567890_abc123
```

**Response:** `200 OK`
```json
{
  "message": "Webhook deleted successfully",
  "webhook_id": "wh_1234567890_abc123"
}
```

**Error Responses:**
- `404 Not Found` - Webhook not found

---

### GET `/api/webhooks`

List all registered webhooks.

**Request:**
```bash
GET /api/webhooks
```

**Response:** `200 OK`
```json
{
  "webhooks": [
    {
      "webhook_id": "wh_1234567890_abc123",
      "callback_url": "https://your-agent.com/webhook/proof-complete",
      "created_at": "2025-10-29T12:00:00.000Z",
      "deliveries": 5
    }
  ]
}
```

---

### Webhook Payload Format

When a proof completes, your webhook endpoint will receive a POST request:

**Headers:**
```
Content-Type: application/json
X-zkX402-Event: proof.completed
X-zkX402-Webhook-ID: wh_1234567890_abc123
```

**Body:**
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

**Webhook Implementation Requirements:**

1. Your webhook endpoint must accept POST requests
2. Return a 2xx status code to acknowledge receipt
3. Respond within 10 seconds (timeout)
4. Handle retries gracefully (idempotent)

**Security Recommendations:**

- Use HTTPS for webhook URLs
- Verify the `X-zkX402-Webhook-ID` header matches your registered webhook
- Implement webhook signature verification (future feature)
- Validate the timestamp to prevent replay attacks

---

## Data Models

### Policy Object

```typescript
interface Policy {
  id: string;                    // Unique identifier
  name: string;                  // Human-readable name
  description: string;           // Policy description
  category: string;              // financial | trust | access | multi-factor | neural
  complexity: string;            // simple | medium | advanced
  operations: number;            // Number of ONNX operations
  avg_proof_time_ms: number;     // Average proof time in ms
  price_usdc: string;            // Price in USDC (decimal)
  price_atomic: string;          // Price in atomic units
  schema: Schema;                // Input/output schema
  example: Examples;             // Example requests
  use_case: string;              // Recommended use case
  endpoint: string;              // Direct authorization URL
  schema_url: string;            // Detailed schema URL
}
```

### Schema Object

```typescript
interface Schema {
  inputs: {
    [key: string]: {
      type: string;              // int8 | int16 | int32 | float32
      description: string;       // Input description
      required: boolean;         // Whether input is required
      validation?: {             // Optional validation rules
        type: string;
        minimum?: number;
        maximum?: number;
      };
    };
  };
  output: {
    type: string;                // Output type
    description: string;         // Output description
    values?: {                   // Output value meanings
      [key: string]: string;
    };
  };
}
```

### Examples Object

```typescript
interface Examples {
  approve: {
    description: string;
    inputs: Record<string, number>;
    expected_output?: number;
  };
  deny: {
    description: string;
    inputs: Record<string, number>;
    expected_output?: number;
  };
}
```

---

## Error Responses

### Standard Error Format

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

### Common Error Codes

#### 400 Bad Request

**Missing Input:**
```json
{
  "error": "Missing required input: balance",
  "code": "MISSING_INPUT",
  "required_inputs": ["amount", "balance"]
}
```

**Invalid Input Type:**
```json
{
  "error": "Invalid input type for 'amount': expected int32, got string",
  "code": "INVALID_INPUT_TYPE"
}
```

#### 402 Payment Required

```json
{
  "error": "Insufficient payment",
  "code": "PAYMENT_REQUIRED",
  "required": "0.001 USDC",
  "provided": "0",
  "payment_address": "0x...",
  "network": "base"
}
```

#### 404 Not Found

```json
{
  "error": "Policy not found",
  "code": "POLICY_NOT_FOUND",
  "policy_id": "invalid_policy",
  "available_policies_url": "/api/policies"
}
```

#### 500 Internal Server Error

```json
{
  "error": "Proof generation failed",
  "code": "PROOF_GENERATION_FAILED",
  "details": "Tensor size exceeds maximum limit (1024 elements)",
  "model_id": "complex_policy"
}
```

---

## Rate Limiting

**Current Limits:**
- No rate limiting implemented yet
- Future: 100 requests/minute per IP
- Future: 1000 proof generations/hour per API key

---

## Versioning

**Current Version:** 1.3.0

**Version History:**
- `1.3.0` (2025-10-29): Added agent API endpoints (`/api/policies`, `/api/policies/:id/schema`)
- `1.2.0`: Enhanced `.well-known/x402` with capabilities
- `1.1.0`: Added 14 curated authorization policies
- `1.0.0`: Initial release with basic proof generation

**Breaking Changes:** None planned. New features will be added in backward-compatible manner.

---

## SDK Support

### Python SDK (Coming Soon)

```python
from zkx402 import Client

client = Client("https://your-server.com")
policies = client.list_policies()
proof = client.generate_proof("simple_threshold", {
    "amount": 5000,
    "balance": 10000
})
```

### JavaScript SDK (Coming Soon)

```javascript
import { zkX402Client } from '@zkx402/sdk';

const client = new zkX402Client('https://your-server.com');
const policies = await client.listPolicies();
const proof = await client.generateProof('simple_threshold', {
  amount: 5000,
  balance: 10000
});
```

---

## Support

- **Documentation**: https://github.com/hshadab/zkx402
- **Issues**: https://github.com/hshadab/zkx402/issues
- **Integration Guide**: See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md)
