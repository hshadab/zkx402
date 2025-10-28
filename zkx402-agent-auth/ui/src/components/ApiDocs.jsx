import React, { useState } from 'react'

const API_ENDPOINTS = [
  {
    method: 'POST',
    path: '/api/generate-proof',
    description: 'Generate a verifiable zkML proof for transaction authorization',
    color: 'accent-green',
    requestBody: {
      model: 'simple_auth | neural_auth | comparison_demo | tensor_ops_demo | matmult_1d_demo',
      inputs: {
        amount: 500,
        balance: 10000,
        velocity_1h: 200,
        velocity_24h: 1500,
        vendor_trust: 80
      }
    },
    responseBody: {
      approved: true,
      output: 85,
      verification: true,
      proofSize: '18.5 KB',
      provingTime: '1247ms',
      verificationTime: '312ms',
      operations: 47,
      zkmlProof: {
        commitment: '2a4f',
        response: '2f',
        evaluation: '55'
      }
    }
  },
  {
    method: 'GET',
    path: '/api/models',
    description: 'List all available ONNX models',
    color: 'accent-blue',
    responseBody: {
      models: [
        {
          id: 'simple_auth',
          file: 'simple_auth.onnx',
          description: 'Simple rule-based authorization',
          inputCount: 5,
          available: true
        },
        {
          id: 'neural_auth',
          file: 'neural_auth.onnx',
          description: 'Neural network authorization',
          inputCount: 5,
          available: true
        }
      ]
    }
  },
  {
    method: 'GET',
    path: '/api/health',
    description: 'Health check endpoint for monitoring',
    color: 'accent-purple',
    responseBody: {
      status: 'ok',
      timestamp: '2025-01-27T12:00:00.000Z',
      modelsDir: '/app/policy-examples/onnx',
      modelsAvailable: 5
    }
  },
  {
    method: 'GET',
    path: '/api/validate-models',
    description: 'Validate that all ONNX model files exist',
    color: 'accent-green',
    responseBody: {
      valid: true,
      models: [
        {
          id: 'simple_auth',
          file: 'simple_auth.onnx',
          exists: true,
          path: '/app/policy-examples/onnx/simple_auth.onnx'
        }
      ]
    }
  },
  {
    method: 'POST',
    path: '/api/upload-model',
    description: 'Upload a custom ONNX model (multipart/form-data)',
    color: 'accent-blue',
    requestBody: 'FormData with "model" field (ONNX file, max 100MB)',
    responseBody: {
      success: true,
      message: 'Model uploaded successfully',
      filename: 'custom_model.onnx',
      path: '/app/policy-examples/onnx/custom_model.onnx',
      size: 12345
    }
  },
  {
    method: 'DELETE',
    path: '/api/models/:modelId',
    description: 'Delete a custom model (built-in models are protected)',
    color: 'accent-purple',
    responseBody: {
      success: true,
      message: 'Model "custom_model" deleted successfully'
    }
  },
  {
    method: 'GET',
    path: '/api/models/:modelId/download',
    description: 'Download an ONNX model file',
    color: 'accent-green',
    responseBody: 'Binary ONNX file download'
  },
  {
    method: 'POST',
    path: '/api/models/:modelId/validate',
    description: 'Validate a model structure (checks file existence)',
    color: 'accent-blue',
    responseBody: {
      valid: true,
      message: 'Model is valid',
      modelId: 'simple_auth'
    }
  }
]

const CODE_EXAMPLES = {
  javascript: {
    name: 'JavaScript (axios)',
    code: `import axios from 'axios';

// Generate proof
const response = await axios.post('http://localhost:3001/api/generate-proof', {
  model: 'neural_auth',
  inputs: {
    amount: 500,
    balance: 10000,
    velocity_1h: 200,
    velocity_24h: 1500,
    vendor_trust: 80
  }
});

console.log('Approved:', response.data.approved);
console.log('Proof size:', response.data.proofSize);
console.log('Verification:', response.data.verification);`
  },
  curl: {
    name: 'cURL',
    code: `# Generate proof
curl -X POST http://localhost:3001/api/generate-proof \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "neural_auth",
    "inputs": {
      "amount": 500,
      "balance": 10000,
      "velocity_1h": 200,
      "velocity_24h": 1500,
      "vendor_trust": 80
    }
  }'

# List models
curl http://localhost:3001/api/models

# Health check
curl http://localhost:3001/api/health`
  },
  python: {
    name: 'Python (requests)',
    code: `import requests

# Generate proof
response = requests.post('http://localhost:3001/api/generate-proof', json={
    'model': 'neural_auth',
    'inputs': {
        'amount': 500,
        'balance': 10000,
        'velocity_1h': 200,
        'velocity_24h': 1500,
        'vendor_trust': 80
    }
})

data = response.json()
print(f"Approved: {data['approved']}")
print(f"Proof size: {data['proofSize']}")
print(f"Verification: {data['verification']}")`
  }
}

export default function ApiDocs() {
  const [selectedEndpoint, setSelectedEndpoint] = useState(0)
  const [selectedExample, setSelectedExample] = useState('javascript')
  const [copiedEndpoint, setCopiedEndpoint] = useState(null)

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text)
    setCopiedEndpoint(index)
    setTimeout(() => setCopiedEndpoint(null), 2000)
  }

  return (
    <div className="space-y-8">
      {/* Overview */}
      <div className="card">
        <h2 className="card-header">API Documentation</h2>
        <div className="space-y-4 text-gray-300">
          <p>
            The zkX402 API provides RESTful endpoints for generating verifiable zkML proofs
            for transaction authorization. All responses are in JSON format.
          </p>
          <div className="bg-dark-700 p-4 rounded-lg">
            <div className="text-sm text-gray-400 mb-2">Base URL (Development)</div>
            <code className="text-accent-green">http://localhost:3001</code>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <div className="text-sm text-gray-400 mb-2">Base URL (Production)</div>
            <code className="text-accent-green">https://your-app.onrender.com</code>
          </div>
        </div>
      </div>

      {/* Endpoints */}
      <div className="card">
        <h2 className="card-header">API Endpoints</h2>
        <div className="space-y-4">
          {API_ENDPOINTS.map((endpoint, index) => (
            <div
              key={index}
              className={`border-2 rounded-lg p-4 cursor-pointer transition-all ${
                selectedEndpoint === index
                  ? `border-${endpoint.color} bg-${endpoint.color}/10`
                  : 'border-dark-600 bg-dark-700 hover:border-dark-500'
              }`}
              onClick={() => setSelectedEndpoint(index)}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    endpoint.method === 'GET' ? 'bg-blue-500/20 text-blue-400' :
                    endpoint.method === 'POST' ? 'bg-green-500/20 text-green-400' :
                    'bg-red-500/20 text-red-400'
                  }`}>
                    {endpoint.method}
                  </span>
                  <code className="text-white font-mono text-sm">{endpoint.path}</code>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    copyToClipboard(endpoint.path, index)
                  }}
                  className="text-xs text-gray-400 hover:text-white transition-colors"
                >
                  {copiedEndpoint === index ? '✓ Copied' : 'Copy'}
                </button>
              </div>
              <p className="text-sm text-gray-400">{endpoint.description}</p>

              {selectedEndpoint === index && (
                <div className="mt-4 space-y-4">
                  {endpoint.requestBody && (
                    <div>
                      <div className="text-xs text-gray-500 mb-2">REQUEST BODY</div>
                      <pre className="bg-dark-900 p-3 rounded overflow-x-auto text-xs">
                        <code className="text-gray-300">
                          {typeof endpoint.requestBody === 'string'
                            ? endpoint.requestBody
                            : JSON.stringify(endpoint.requestBody, null, 2)}
                        </code>
                      </pre>
                    </div>
                  )}
                  {endpoint.responseBody && (
                    <div>
                      <div className="text-xs text-gray-500 mb-2">RESPONSE BODY</div>
                      <pre className="bg-dark-900 p-3 rounded overflow-x-auto text-xs">
                        <code className="text-gray-300">
                          {typeof endpoint.responseBody === 'string'
                            ? endpoint.responseBody
                            : JSON.stringify(endpoint.responseBody, null, 2)}
                        </code>
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Code Examples */}
      <div className="card">
        <h2 className="card-header">Code Examples</h2>
        <div className="space-y-4">
          <div className="flex gap-2">
            {Object.entries(CODE_EXAMPLES).map(([key, example]) => (
              <button
                key={key}
                onClick={() => setSelectedExample(key)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedExample === key
                    ? 'bg-accent-green text-dark-900'
                    : 'bg-dark-700 text-gray-400 hover:bg-dark-600'
                }`}
              >
                {example.name}
              </button>
            ))}
          </div>
          <pre className="bg-dark-900 p-4 rounded-lg overflow-x-auto">
            <code className="text-sm text-gray-300">
              {CODE_EXAMPLES[selectedExample].code}
            </code>
          </pre>
        </div>
      </div>

      {/* Integration Notes */}
      <div className="card">
        <h2 className="card-header">Integration Notes</h2>
        <div className="space-y-4 text-gray-300">
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-green font-semibold mb-2">Input Scaling</h3>
            <p className="text-sm">
              All monetary values are scaled by 100 (e.g., 500 = $5.00). This is required
              because JOLT Atlas uses integer-only operations for zkML proofs.
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-blue font-semibold mb-2">Proof Generation Time</h3>
            <p className="text-sm">
              Expect 0.5-3 seconds for proof generation depending on model complexity.
              Simple rule-based models (~0.5s), neural networks (~1.5s), complex models (~3s).
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-purple font-semibold mb-2">Verification</h3>
            <p className="text-sm">
              The <code className="text-accent-green">verification</code> field indicates whether
              the proof cryptographically verifies. This ensures the authorization decision was
              computed correctly and hasn't been tampered with.
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-green font-semibold mb-2">x402 Integration</h3>
            <p className="text-sm mb-2">
              To integrate with x402 payment protocol, include the zkML proof in the
              x402 payment header or response body:
            </p>
            <pre className="bg-dark-900 p-2 rounded text-xs">
              <code className="text-gray-300">
{`X402-Authorization-Proof: {
  "approved": true,
  "verification": true,
  "zkmlProof": {...}
}`}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  )
}
