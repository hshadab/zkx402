# zkX402 External API Server

Production-ready REST API for zkX402 Agent Authorization with JOLT Atlas.

## Features

- **RESTful API**: Clean, versioned API endpoints
- **Rate Limiting**: Protect against abuse
- **Request Validation**: Input validation with express-validator
- **Structured Logging**: Winston logger with JSON output
- **Batch Processing**: Generate multiple proofs in parallel
- **CORS Support**: Configurable cross-origin requests
- **Security**: Helmet.js security headers
- **Request Tracking**: UUID-based request tracking

## Quick Start

### Installation

```bash
cd zkx402-agent-auth/api-server
npm install
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your configuration
```

### Run

```bash
# Development
npm run dev

# Production
npm start
```

Server will start on `http://localhost:4000`

## API Documentation

### Base URL

```
http://localhost:4000/api/v1
```

### Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-10-28T12:00:00.000Z",
  "uptime": 3600.5,
  "modelsAvailable": 5
}
```

#### GET /models

List available authorization models.

**Response:**
```json
{
  "models": [
    {
      "id": "simple_auth",
      "file": "simple_auth.onnx",
      "description": "Simple rule-based authorization",
      "inputCount": 5,
      "estimatedTime": "700ms",
      "available": true
    }
  ],
  "count": 5
}
```

#### POST /proof

Generate a single authorization proof.

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

**Response:**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "approved": true,
  "output": 100,
  "verification": true,
  "proofSize": "15.2 KB",
  "provingTime": "700ms",
  "verificationTime": "45ms",
  "operations": 21,
  "zkmlProof": {
    "commitment": "0x...",
    "response": "0x...",
    "evaluation": "0x..."
  },
  "metadata": {
    "model": "simple_auth",
    "inputs": { ... },
    "timestamp": "2025-10-28T12:00:00.000Z",
    "duration": "750ms"
  }
}
```

#### POST /proof/batch

Generate multiple proofs in parallel (max 10 per request).

**Request:**
```json
{
  "requests": [
    {
      "model": "simple_auth",
      "inputs": {
        "amount": "50",
        "balance": "1000",
        "velocity_1h": "20",
        "velocity_24h": "100",
        "vendor_trust": "80"
      }
    },
    {
      "model": "neural_auth",
      "inputs": {
        "amount": "100",
        "balance": "2000",
        "velocity_1h": "30",
        "velocity_24h": "150",
        "vendor_trust": "75"
      }
    }
  ]
}
```

**Response:**
```json
{
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "success": true,
      "result": { ... }
    },
    {
      "success": true,
      "result": { ... }
    }
  ]
}
```

## Integration Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_BASE = 'http://localhost:4000/api/v1';

async function authorizeTransaction(amount, balance, velocity_1h, velocity_24h, vendor_trust) {
  try {
    const response = await axios.post(`${API_BASE}/proof`, {
      model: 'simple_auth',
      inputs: {
        amount: amount.toString(),
        balance: balance.toString(),
        velocity_1h: velocity_1h.toString(),
        velocity_24h: velocity_24h.toString(),
        vendor_trust: vendor_trust.toString()
      }
    });

    console.log('Request ID:', response.data.requestId);
    console.log('Approved:', response.data.approved);
    console.log('Proof Size:', response.data.proofSize);

    return response.data;
  } catch (error) {
    console.error('Authorization failed:', error.response?.data || error.message);
    throw error;
  }
}

// Usage
authorizeTransaction(50, 1000, 20, 100, 80)
  .then(result => console.log('Authorization result:', result))
  .catch(error => console.error('Error:', error));
```

### Python

```python
import requests

API_BASE = 'http://localhost:4000/api/v1'

def authorize_transaction(amount, balance, velocity_1h, velocity_24h, vendor_trust):
    response = requests.post(f'{API_BASE}/proof', json={
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
        print(f"Request ID: {result['requestId']}")
        print(f"Approved: {result['approved']}")
        print(f"Proof Size: {result['proofSize']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Usage
authorize_transaction(50, 1000, 20, 100, 80)
```

### cURL

```bash
curl -X POST http://localhost:4000/api/v1/proof \
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

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 4000 | Server port |
| `NODE_ENV` | development | Environment (development/production) |
| `ALLOWED_ORIGINS` | * | CORS allowed origins (comma-separated) |
| `RATE_LIMIT` | 100 | Max requests per 15 minutes |
| `LOG_LEVEL` | info | Logging level (debug/info/warn/error) |

### Rate Limiting

Default: 100 requests per 15 minutes per IP

To adjust:
```bash
RATE_LIMIT=200  # 200 requests per 15 minutes
```

### CORS

By default, all origins are allowed. To restrict:

```bash
ALLOWED_ORIGINS=https://your-app.com,https://another-app.com
```

## Logging

Logs are written to:
- `combined.log` - All logs
- `error.log` - Error logs only
- Console - Development logs

Log format: JSON with timestamps

Example log entry:
```json
{
  "level": "info",
  "message": "Proof generated successfully",
  "timestamp": "2025-10-28T12:00:00.000Z",
  "requestId": "550e8400-e29b-41d4-a716-446655440000",
  "model": "simple_auth",
  "approved": true,
  "duration": "750ms"
}
```

## Error Handling

All errors return this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "requestId": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Common Errors

| Status | Error | Solution |
|--------|-------|----------|
| 400 | Validation failed | Check input format |
| 404 | Not found | Check endpoint URL |
| 429 | Too many requests | Wait and retry |
| 500 | Proof generation failed | Check logs, verify Rust build |

## Performance

- **Single proof**: ~750ms (simple_auth) to ~1.6s (neural_auth)
- **Batch processing**: Parallel execution, ~same time as single
- **Throughput**: ~80 proofs/minute (depends on hardware)

## Security

### Production Checklist

- [ ] Set `NODE_ENV=production`
- [ ] Configure `ALLOWED_ORIGINS`
- [ ] Enable HTTPS (use reverse proxy)
- [ ] Implement API key authentication (if needed)
- [ ] Set up monitoring and alerts
- [ ] Configure log rotation
- [ ] Review rate limits

### Adding API Key Authentication

To add API key auth:

```javascript
// In server.js
const apiKeyMiddleware = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey || apiKey !== process.env.API_KEY_SECRET) {
    return res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid or missing API key'
    });
  }

  next();
};

// Apply to routes
app.use('/api/v1/proof', apiKeyMiddleware);
```

## Testing

```bash
npm test
```

## Deployment

See main [DEPLOYMENT.md](../../DEPLOYMENT.md) for deployment instructions.

## Support

- **Issues**: https://github.com/yourusername/zkx402/issues
- **API Docs**: https://github.com/yourusername/zkx402/blob/main/API_REFERENCE.md

---

**Version**: 1.0.0
**License**: MIT
