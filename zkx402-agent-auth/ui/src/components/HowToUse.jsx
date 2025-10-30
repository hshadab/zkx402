import { Code, Zap, Users, Shield } from 'lucide-react';

export default function HowToUse() {
  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-white mb-4">How to Use zkX402</h1>
        <p className="text-xl text-gray-400">
          Verifiable AI agent authorization in two simple ways
        </p>
      </div>

      {/* For Human Users */}
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-accent-green/20 rounded-lg">
            <Users className="w-6 h-6 text-accent-green" />
          </div>
          <h2 className="text-2xl font-bold text-white">For Human Users (Web Interface)</h2>
        </div>

        <div className="space-y-6 text-gray-300">
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Step 1: Choose a Model</h3>
            <p className="text-gray-400 mb-2">
              Start by selecting one of our 14 authorization models:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4 text-gray-400">
              <li><strong className="text-accent-green">Rule-Based Models</strong> - Simple threshold checks (e.g., "amount &lt; balance")</li>
              <li><strong className="text-accent-blue">Neural Network Models</strong> - ML-based risk scoring</li>
              <li><strong className="text-accent-purple">Hybrid Models</strong> - Combined logic and ML</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Step 2: Test Authorization</h3>
            <p className="text-gray-400">
              Enter sample data (like transaction amount, balance, trust score) and click
              <strong className="text-accent-green"> "Generate ZK Proof"</strong>. The system will:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4 mt-2 text-gray-400">
              <li>Run the model on your inputs</li>
              <li>Generate a zero-knowledge proof (~700ms)</li>
              <li>Show you whether the request would be approved or denied</li>
              <li>Display the cryptographic proof details</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Step 3: View Results</h3>
            <p className="text-gray-400">
              The proof shows that the authorization decision is correct <strong>without revealing your private data</strong>.
              You can verify the proof independently and see performance metrics like proof size and verification time.
            </p>
          </div>
        </div>
      </div>

      {/* For AI Agents */}
      <div className="card">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-3 bg-accent-blue/20 rounded-lg">
            <Code className="w-6 h-6 text-accent-blue" />
          </div>
          <h2 className="text-2xl font-bold text-white">For AI Agents (API)</h2>
        </div>

        <div className="space-y-6 text-gray-300">
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">What Agents Can Do</h3>
            <p className="text-gray-400 mb-4">
              AI agents can use zkX402 to prove they're authorized to perform actions without revealing their
              private data (like wallet balances, transaction history, or risk scores).
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-3">How It Works for Agents</h3>
            <div className="space-y-4">
              <div className="bg-dark-700 p-4 rounded-lg">
                <h4 className="font-semibold text-accent-green mb-2">Step 1: Discover the Service</h4>
                <p className="text-sm text-gray-400 mb-2">Agents call the x402 discovery endpoint:</p>
                <code className="block bg-dark-900 p-3 rounded text-xs overflow-x-auto">
                  GET /.well-known/x402
                </code>
                <p className="text-sm text-gray-400 mt-2">
                  This returns available models, payment info, and pricing.
                </p>
              </div>

              <div className="bg-dark-700 p-4 rounded-lg">
                <h4 className="font-semibold text-accent-blue mb-2">Step 2: Request Authorization</h4>
                <p className="text-sm text-gray-400 mb-2">
                  Agents send their data and pay with Base USDC:
                </p>
                <code className="block bg-dark-900 p-3 rounded text-xs overflow-x-auto whitespace-pre">
{`POST /x402/authorize/:modelId
Content-Type: application/json
X-Payment: <base64-encoded-payment-proof>

{
  "inputs": {
    "amount": "5000",
    "balance": "10000",
    "risk_score": "0.85"
  }
}`}
                </code>
              </div>

              <div className="bg-dark-700 p-4 rounded-lg">
                <h4 className="font-semibold text-accent-purple mb-2">Step 3: Get ZK Proof</h4>
                <p className="text-sm text-gray-400 mb-2">
                  The agent receives a zero-knowledge proof showing the authorization decision:
                </p>
                <code className="block bg-dark-900 p-3 rounded text-xs overflow-x-auto whitespace-pre">
{`{
  "approved": true,
  "zkml_proof": {
    "commitment": "0x...",
    "response": "0x...",
    "evaluation": "0x..."
  },
  "verification_time": "70ms"
}`}
                </code>
                <p className="text-sm text-gray-400 mt-2">
                  The agent can now prove it was authorized <strong>without revealing its private inputs</strong>.
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Key Benefits for Agents</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-start gap-3 bg-dark-700 p-4 rounded-lg">
                <Shield className="w-5 h-5 text-accent-green flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-white mb-1">Private</h4>
                  <p className="text-sm text-gray-400">
                    Prove authorization without revealing wallet balances or transaction history
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 bg-dark-700 p-4 rounded-lg">
                <Zap className="w-5 h-5 text-accent-blue flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-white mb-1">Fast</h4>
                  <p className="text-sm text-gray-400">
                    Get authorization proofs in ~700ms with instant verification
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 bg-dark-700 p-4 rounded-lg">
                <Code className="w-5 h-5 text-accent-purple flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-white mb-1">Verifiable</h4>
                  <p className="text-sm text-gray-400">
                    Anyone can verify the proof cryptographically without seeing private data
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3 bg-dark-700 p-4 rounded-lg">
                <Users className="w-5 h-5 text-accent-green flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-white mb-1">Flexible</h4>
                  <p className="text-sm text-gray-400">
                    Upload custom ONNX models or use 14 pre-built authorization models
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Links */}
      <div className="bg-gradient-to-r from-dark-800 to-dark-700 rounded-lg p-6 border border-accent-green/20">
        <h3 className="text-lg font-semibold text-white mb-4">Ready to Get Started?</h3>
        <div className="flex flex-col sm:flex-row gap-4">
          <a
            href="#simulator"
            onClick={() => window.dispatchEvent(new Event('setActiveTab-simulator'))}
            className="flex-1 px-6 py-3 bg-gradient-to-r from-accent-green to-accent-blue hover:from-accent-green/90 hover:to-accent-blue/90 text-dark-900 font-semibold rounded-lg transition-all text-center"
          >
            Try the Web Interface
          </a>
          <a
            href="#api"
            onClick={() => window.dispatchEvent(new Event('setActiveTab-api'))}
            className="flex-1 px-6 py-3 bg-dark-600 hover:bg-dark-500 text-white font-semibold rounded-lg transition-colors border border-dark-500 text-center"
          >
            View API Documentation
          </a>
        </div>
      </div>
    </div>
  );
}
