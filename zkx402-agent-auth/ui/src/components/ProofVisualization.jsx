import React, { useState } from 'react'

export default function ProofVisualization({ proofData }) {
  const [showRawProof, setShowRawProof] = useState(false)

  const { approved, output, verification, inputs, modelType } = proofData

  return (
    <div className="card">
      <h2 className="card-header">Zero-Knowledge Proof Result</h2>

      <div className="space-y-6">
        {/* Authorization Result */}
        <div className="bg-dark-700 p-6 rounded-lg border border-dark-600">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Authorization Status</h3>
            <span className={`status-badge ${approved ? 'status-success' : 'status-error'}`}>
              {approved ? '✓ Approved' : '✗ Denied'}
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-400 mb-1">Model Type</p>
              <p className="text-white font-mono">{modelType}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Output Value</p>
              <p className="text-accent-green font-mono">{output}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Verification</p>
              <p className={`font-mono ${verification ? 'text-accent-green' : 'text-red-400'}`}>
                {verification ? 'Valid ✓' : 'Invalid ✗'}
              </p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Privacy</p>
              <p className="text-accent-green font-mono">Zero-Knowledge</p>
            </div>
          </div>
        </div>

        {/* Input Summary */}
        <div className="bg-dark-700 p-6 rounded-lg border border-dark-600">
          <h3 className="text-lg font-semibold mb-4">Transaction Details (Hidden from Verifier)</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 text-sm">
            <div>
              <p className="text-gray-400 mb-1">Amount</p>
              <p className="text-white font-mono">${(inputs.amount / 100).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Balance</p>
              <p className="text-white font-mono">${(inputs.balance / 100).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Velocity 1h</p>
              <p className="text-white font-mono">${(inputs.velocity_1h / 100).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Velocity 24h</p>
              <p className="text-white font-mono">${(inputs.velocity_24h / 100).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-gray-400 mb-1">Trust Score</p>
              <p className="text-white font-mono">{inputs.vendor_trust}/100</p>
            </div>
          </div>
        </div>

        {/* Proof Details */}
        <div className="bg-dark-700 p-6 rounded-lg border border-dark-600">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">zkML Proof Details</h3>
            <button
              onClick={() => setShowRawProof(!showRawProof)}
              className="text-sm text-accent-green hover:underline"
            >
              {showRawProof ? 'Hide' : 'View'} Raw Proof
            </button>
          </div>

          {showRawProof && (
            <div className="mt-4">
              <pre className="bg-dark-900 p-4 rounded overflow-x-auto text-xs text-gray-300 border border-dark-600">
                {JSON.stringify(proofData, null, 2)}
              </pre>
            </div>
          )}

          <div className="text-sm text-gray-400 space-y-2">
            <p>✓ Proof generated using JOLT Atlas zkML</p>
            <p>✓ Private inputs never revealed to verifier</p>
            <p>✓ Cryptographically verifiable authorization decision</p>
            <p>✓ Enhanced with Greater, Less, Slice, Identity operations</p>
          </div>
        </div>
      </div>
    </div>
  )
}
