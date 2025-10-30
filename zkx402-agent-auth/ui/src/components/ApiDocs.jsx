import React, { useState } from 'react'

const API_ENDPOINTS = [
  {
    method: 'GET',
    path: '/.well-known/x402',
    description: 'x402 protocol service discovery endpoint - lists capabilities, pricing, and all available policies',
    color: 'accent-purple',
    responseBody: {
      service: 'zkX402 Privacy-Preserving Authorization for AI Agents',
      version: '1.3.0',
      status: 'production',
      capabilities: {
        zkml_proofs: true,
        max_model_params: 1024,
        max_operations: 100,
        supported_input_types: ['int8', 'int16', 'int32', 'float32'],
        proof_time_range: '0.5s - 9s'
      },
      pricing: {
        currency: 'USDC',
        network: 'base'
      },
      endpoints: {
        list_policies: '/api/policies',
        get_policy_schema: '/api/policies/:id/schema',
        generate_proof: '/api/generate-proof'
      },
      pre_built_policies: '... 14 curated authorization policies ...'
    }
  },
  {
    method: 'GET',
    path: '/api/policies',
    description: 'List all 14 curated authorization policies with metadata for autonomous agent discovery',
    color: 'accent-blue',
    responseBody: {
      version: '1.3.0',
      total_policies: 14,
      policies: [
        {
          id: 'simple_threshold',
          name: 'Simple Threshold Check',
          description: 'Approve if amount < balance',
          category: 'financial',
          complexity: 'simple',
          operations: 2,
          avg_proof_time_ms: 500,
          price_usdc: '0.0001',
          schema: {
            inputs: {
              amount: { type: 'int32', description: 'Transaction amount', required: true },
              balance: { type: 'int32', description: 'Account balance', required: true }
            },
            output: { type: 'int32', description: '1 = approved, 0 = denied' }
          },
          example: {
            approve: { amount: 5000, balance: 10000 },
            deny: { amount: 15000, balance: 10000 }
          },
          endpoint: '/x402/authorize/simple_threshold',
          schema_url: '/api/policies/simple_threshold/schema'
        }
      ]
    }
  },
  {
    method: 'GET',
    path: '/api/policies/:id/schema',
    description: 'Get detailed schema for a specific authorization policy',
    color: 'accent-green',
    responseBody: {
      policy_id: 'simple_threshold',
      name: 'Simple Threshold Check',
      description: 'Approve if amount < balance',
      category: 'financial',
      schema: {
        inputs: {
          amount: {
            type: 'int32',
            description: 'Transaction amount to authorize',
            required: true,
            validation: { type: 'integer', minimum: 0 }
          },
          balance: {
            type: 'int32',
            description: 'Current account balance',
            required: true,
            validation: { type: 'integer', minimum: 0 }
          }
        },
        output: {
          type: 'int32',
          description: '1 = approved, 0 = denied',
          values: { 1: 'approved', 0: 'denied' }
        }
      },
      examples: {
        approve: { inputs: { amount: 5000, balance: 10000 }, expected_output: 1 },
        deny: { inputs: { amount: 15000, balance: 10000 }, expected_output: 0 }
      },
      pricing: { price_usdc: '0.0001', currency: 'USDC', network: 'base' },
      performance: { operations: 2, avg_proof_time: '~0.5s', complexity: 'simple' }
    }
  },
  {
    method: 'POST',
    path: '/api/generate-proof',
    description: 'Generate a verifiable zkML proof for transaction authorization',
    color: 'accent-green',
    requestBody: {
      model: 'simple_threshold | percentage_limit | vendor_trust | velocity_1h | ...',
      inputs: {
        amount: 5000,
        balance: 10000
      }
    },
    responseBody: {
      approved: true,
      output: 1,
      proof: '0x1a2b3c4d...',
      verification: {
        verified: true,
        verifier: 'jolt_atlas_v1',
        timestamp: '2025-10-29T12:34:56.789Z'
      },
      proving_time: '732ms',
      proof_size: '15360',
      model_type: 'simple_threshold',
      inputs_hash: '0xabc123...',
      payment: {
        required: '100',
        currency: 'USDC',
        network: 'base'
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
    name: 'JavaScript (fetch)',
    code: `// Agent Discovery & Proof Generation Workflow
const BASE_URL = 'http://localhost:3001';

// 1. Service discovery (x402 protocol)
const discovery = await fetch(\`\${BASE_URL}/.well-known/x402\`)
  .then(r => r.json());

console.log('Service:', discovery.service);
console.log('Available policies:', discovery.pre_built_policies.length);

// 2. List all policies
const policies = await fetch(\`\${BASE_URL}/api/policies\`)
  .then(r => r.json());

// 3. Select policy by use case
const policy = policies.policies.find(p => p.id === 'simple_threshold');
console.log('Using:', policy.name);

// 4. Generate proof
const proof = await fetch(\`\${BASE_URL}/api/generate-proof\`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: policy.id,
    inputs: { amount: 5000, balance: 10000 }
  })
}).then(r => r.json());

console.log('Approved:', proof.approved);
console.log('Proving time:', proof.proving_time);
console.log('Verified:', proof.verification.verified);`
  },
  curl: {
    name: 'cURL',
    code: `# 1. Service discovery
curl http://localhost:3001/.well-known/x402

# 2. List all policies
curl http://localhost:3001/api/policies

# 3. Get policy schema
curl http://localhost:3001/api/policies/simple_threshold/schema

# 4. Generate proof
curl -X POST http://localhost:3001/api/generate-proof \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": "5000",
      "balance": "10000"
    }
  }'

# 5. Test with multi-factor policy
curl -X POST http://localhost:3001/api/generate-proof \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "multi_factor",
    "inputs": {
      "amount": "5000",
      "balance": "100000",
      "spent_24h": "20000",
      "limit_24h": "50000",
      "vendor_trust": "75",
      "min_trust": "50"
    }
  }'`
  },
  python: {
    name: 'Python (requests)',
    code: `import requests

BASE_URL = 'http://localhost:3001'

# 1. Service discovery (x402 protocol compliance)
discovery = requests.get(f'{BASE_URL}/.well-known/x402').json()
print(f"Service: {discovery['service']}")
print(f"Policies: {len(discovery['pre_built_policies'])}")

# 2. List all policies with metadata
policies = requests.get(f'{BASE_URL}/api/policies').json()

# 3. Select policy by category
financial_policies = [
    p for p in policies['policies']
    if p['category'] == 'financial'
]
policy = financial_policies[0]
print(f"Using: {policy['name']} - {policy['description']}")

# 4. Generate zero-knowledge proof
proof = requests.post(
    f'{BASE_URL}/api/generate-proof',
    json={
        'model': policy['id'],
        'inputs': {
            'amount': '5000',
            'balance': '10000'
        }
    }
).json()

if proof['approved']:
    print(f"✓ Authorization approved")
    print(f"  Proof: {proof['proof'][:32]}...")
    print(f"  Time: {proof['proving_time']}")
    print(f"  Verified: {proof['verification']['verified']}")`
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
            <strong>zkX402 v1.3.0</strong> - Privacy-preserving authorization for autonomous x402 agents.
            Generate zero-knowledge proofs for 14 curated authorization policies without revealing private data.
            All responses are in JSON format with x402 protocol compliance.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-dark-700 p-4 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">Base URL (Local)</div>
              <code className="text-accent-green">http://localhost:3001</code>
            </div>
            <div className="bg-dark-700 p-4 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">Start Here</div>
              <code className="text-accent-purple">GET /.well-known/x402</code>
            </div>
          </div>
          <div className="bg-gradient-to-r from-accent-purple/10 to-accent-blue/10 border border-accent-purple/30 p-4 rounded-lg">
            <div className="flex items-start gap-3">
              <span className="text-2xl">🤖</span>
              <div>
                <h3 className="text-accent-purple font-semibold mb-1">Built for Autonomous Agents</h3>
                <p className="text-sm text-gray-400">
                  Agents can discover policies, understand schemas, and generate proofs programmatically.
                  Start with <code className="text-accent-green">/.well-known/x402</code> for service discovery.
                </p>
              </div>
            </div>
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
            <h3 className="text-accent-purple font-semibold mb-2">x402 Protocol Compliance</h3>
            <p className="text-sm mb-2">
              This service implements the x402 protocol for autonomous agent authorization.
              Start with <code className="text-accent-green">/.well-known/x402</code> for
              service discovery, then use <code className="text-accent-green">/api/policies</code> to
              browse available authorization policies.
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-green font-semibold mb-2">Agent Discovery Workflow</h3>
            <p className="text-sm">
              <strong>1.</strong> Fetch <code className="text-accent-green">/.well-known/x402</code> to
              discover capabilities and available policies<br/>
              <strong>2.</strong> Use <code className="text-accent-green">/api/policies</code> to
              list all 14 curated policies with metadata<br/>
              <strong>3.</strong> Get detailed schema via <code className="text-accent-green">/api/policies/:id/schema</code><br/>
              <strong>4.</strong> Generate proofs using <code className="text-accent-green">/api/generate-proof</code>
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-blue font-semibold mb-2">14 Curated Authorization Policies</h3>
            <p className="text-sm mb-2">
              We provide 14 pre-built policies covering common authorization use cases:
            </p>
            <ul className="text-xs space-y-1 list-disc list-inside">
              <li><strong>Financial:</strong> simple_threshold, percentage_limit, velocity_1h, velocity_24h, daily_limit</li>
              <li><strong>Trust:</strong> vendor_trust</li>
              <li><strong>Access:</strong> age_gate</li>
              <li><strong>Multi-Factor:</strong> multi_factor, composite_scoring</li>
              <li><strong>Neural:</strong> risk_neural (ML-based fraud detection)</li>
              <li><strong>Test Ops:</strong> test_less, test_identity, test_clip, test_slice</li>
            </ul>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-green font-semibold mb-2">JOLT Atlas Constraints</h3>
            <p className="text-sm">
              <strong>Max tensor size:</strong> 1,024 elements<br/>
              <strong>Max operations:</strong> ~100 per model<br/>
              <strong>Integer-only:</strong> All values must be integers (use scaling for decimals)<br/>
              <strong>Proof times:</strong> Simple (0.5-1.5s), Medium (1.5-4s), Complex (4-9s)
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-blue font-semibold mb-2">Verification & Privacy</h3>
            <p className="text-sm">
              The <code className="text-accent-green">verification.verified</code> field confirms
              the proof is cryptographically valid. This ensures the authorization decision was
              computed correctly without revealing private inputs (balance, trust scores, etc.).
            </p>
          </div>
          <div className="bg-dark-700 p-4 rounded-lg">
            <h3 className="text-accent-purple font-semibold mb-2">Documentation</h3>
            <p className="text-sm">
              <strong>AGENT_INTEGRATION.md:</strong> Comprehensive integration guide with examples<br/>
              <strong>API_REFERENCE.md:</strong> Complete API reference with all endpoints<br/>
              <strong>GitHub:</strong> <a href="https://github.com/hshadab/zkx402" className="text-accent-green hover:underline">https://github.com/hshadab/zkx402</a>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
