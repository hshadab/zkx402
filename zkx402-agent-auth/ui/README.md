# zkX402 Agent Authorization UI

Web interface for demonstrating zero-knowledge machine learning authorization using JOLT Atlas.

## Features

- **Dark-themed interface** inspired by rugdetector and zkml-erc8004
- **Real-time proof generation** visualization
- **Multiple model support**: Rule-based, Neural Network, and demo models
- **Interactive simulator** with adjustable input parameters
- **Performance metrics** display
- **Zero-knowledge proof verification**
- **Analytics dashboard** with real-time usage tracking, revenue metrics, and model performance statistics
- **Webhook support** for async proof completion notifications

## Tech Stack

**Frontend:**
- React 18 with Vite
- Tailwind CSS for styling
- Axios for API calls
- JetBrains Mono font

**Backend:**
- Express.js API server
- ONNX model loading
- JOLT Atlas proof generation (simulated)

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Generate ONNX Models

```bash
cd ../policy-examples/onnx
python3 create_demo_models.py
```

### 3. Start Development Server

```bash
npm run dev
```

This starts both:
- Frontend: http://localhost:3000
- Backend API: http://localhost:3001

## Project Structure

```
ui/
├── public/              # Static assets
├── src/
│   ├── components/      # React components
│   │   ├── Header.jsx
│   │   ├── ModelSelector.jsx
│   │   ├── AuthorizationSimulator.jsx
│   │   ├── ProofVisualization.jsx
│   │   ├── PerformanceMetrics.jsx
│   │   ├── Analytics.jsx
│   │   └── ApiDocs.jsx
│   ├── pages/           # Page-level components
│   ├── utils/           # Utility functions
│   ├── api/             # API integration
│   ├── App.jsx          # Main app component
│   ├── main.jsx         # React entry point
│   └── index.css        # Global styles
├── server.js            # Express backend server
├── analytics-manager.js # Analytics tracking module
├── webhook-manager.js   # Webhook notifications
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
├── tailwind.config.js   # Tailwind CSS config
└── README.md            # This file
```

## API Endpoints

### `GET /api/health`
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-01-27T...",
  "modelsDir": "/path/to/models",
  "modelsAvailable": 5
}
```

### `GET /api/models`
List available ONNX models

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

### `POST /api/generate-proof`
Generate zkML proof for authorization

**Request:**
```json
{
  "model": "simple_auth",
  "inputs": {
    "amount": "500",
    "balance": "10000",
    "velocity_1h": "200",
    "velocity_24h": "1500",
    "vendor_trust": "80"
  }
}
```

**Response:**
```json
{
  "approved": true,
  "output": 1,
  "verification": true,
  "proofSize": "15 KB",
  "verificationTime": "70ms",
  "operations": 20,
  "zkmlProof": {
    "commitment": "0x...",
    "response": "0x...",
    "evaluation": "0x..."
  }
}
```

### `GET /api/analytics/stats`
Get comprehensive analytics statistics

**Response:**
```json
{
  "uptime": 3600,
  "totalRequests": 150,
  "requests24h": 45,
  "successRate": 94.5,
  "avgResponseTime": 1200,
  "totalRevenue": "0.0150",
  "revenue24h": "0.0045",
  "verifiedPayments": 12,
  "recentRequests": [...],
  "recentPayments": [...]
}
```

### `GET /api/analytics/models`
Get per-model usage breakdown

**Response:**
```json
{
  "simple_threshold": {
    "totalRequests": 50,
    "successRate": "96.0",
    "paidRequests": 8,
    "avgResponseTime": "850"
  },
  ...
}
```

## Available Models

1. **simple_auth.onnx** - Rule-based authorization
   - Checks: amount < 10% balance, trust > 50, velocity limits
   - ~0.7s proof generation

2. **neural_auth.onnx** - Neural network scoring
   - Architecture: [5] → [8] → [4] → [1]
   - ~1.5s proof generation

3. **comparison_demo.onnx** - Comparison operations
   - Greater, Less, GreaterEqual operations
   - ~0.3s proof generation

4. **tensor_ops_demo.onnx** - Tensor operations
   - Slice, Identity, Reshape operations
   - ~0.3s proof generation

5. **matmult_1d_demo.onnx** - MatMult with 1D outputs
   - Matrix-vector multiplication
   - ~0.4s proof generation

## Development

### Frontend Development

```bash
npm run client
```

Starts Vite dev server on http://localhost:3000

### Backend Development

```bash
npm run server
```

Starts Express server on http://localhost:3001 with nodemon

### Production Build

```bash
npm run build
```

Builds optimized production bundle to `dist/`

## Integration with JOLT Atlas

The current implementation simulates proof generation. To integrate with actual JOLT Atlas:

### Option 1: CLI Execution
```javascript
const { exec } = require('child_process');

exec(`cargo run --example integer_auth_e2e`, (error, stdout) => {
  // Parse proof from stdout
});
```

### Option 2: N-API Bindings
```javascript
const joltAtlas = require('./jolt-bindings');

const proof = joltAtlas.generateProof(modelPath, inputs);
```

### Option 3: WASM Compilation
```javascript
import init, { generate_proof } from './jolt-atlas.wasm';

await init();
const proof = generate_proof(modelPath, inputs);
```

## Styling

Dark theme with accent colors:
- **Primary accent**: `#00ff88` (green)
- **Secondary accent**: `#00ccff` (blue)
- **Tertiary accent**: `#aa66ff` (purple)
- **Background**: `#0a0a0a` to `#2a2a2a` (dark grays)

Font: JetBrains Mono (monospace)

## Performance

Typical performance metrics:
- Frontend load: < 1s
- API response: 50-100ms
- Proof generation: 0.3-1.5s (simulated)
- Total interaction: < 2s

## Known Limitations

1. **Simulated proofs**: Current implementation simulates JOLT Atlas proof generation
2. **No persistence**: Proofs not stored, regenerated each time
3. **Single-threaded**: One proof at a time
4. **No batching**: Cannot process multiple authorizations in parallel

## Future Enhancements

- [ ] Real JOLT Atlas integration via Rust bindings
- [ ] Proof caching and persistence
- [ ] WebSocket for real-time updates
- [ ] Batch proof generation
- [ ] Export proofs as JSON/binary
- [ ] Proof verification UI
- [ ] Historical proof viewer
- [ ] Model upload and management

## Contributing

To add a new model:

1. Create ONNX model in `policy-examples/onnx/`
2. Add model config to `server.js` `MODEL_CONFIGS`
3. Add model option to `ModelSelector.jsx`
4. Test with `POST /api/generate-proof`

## License

MIT License - Built on JOLT Atlas (MIT)

## References

- JOLT Atlas: https://github.com/ICME-Lab/jolt-atlas
- zkX402: https://github.com/hshadab/zkx402
- RugDetector UI: https://github.com/hshadab/rugdetector
<!-- Removed non-MVP reference to ERC-8004 to keep scope focused on zkX402 -->
