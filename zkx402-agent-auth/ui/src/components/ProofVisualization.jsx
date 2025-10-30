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
            <h3 className="text-lg font-semibold">Authorization (Proof) Status</h3>
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
              <p className="text-gray-400 mb-1">Proof Verification</p>
              <p className={`font-mono ${verification ? 'text-accent-green' : 'text-red-400'}`}>
                {verification ? 'Cryptographically Valid ✓' : 'Invalid ✗'}
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
            {inputs.amount !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Amount</p>
                <p className="text-white font-mono">${(inputs.amount / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.balance !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Balance</p>
                <p className="text-white font-mono">${(inputs.balance / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.velocity_1h !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Velocity 1h</p>
                <p className="text-white font-mono">${(inputs.velocity_1h / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.velocity_24h !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Velocity 24h</p>
                <p className="text-white font-mono">${(inputs.velocity_24h / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.spent_1h !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Spent 1h</p>
                <p className="text-white font-mono">${(inputs.spent_1h / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.spent_24h !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Spent 24h</p>
                <p className="text-white font-mono">${(inputs.spent_24h / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.vendor_trust !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Trust Score</p>
                <p className="text-white font-mono">{inputs.vendor_trust}/100</p>
              </div>
            )}
            {inputs.max_percentage !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Max Percentage</p>
                <p className="text-white font-mono">{inputs.max_percentage}%</p>
              </div>
            )}
            {inputs.age !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Age</p>
                <p className="text-white font-mono">{inputs.age} years</p>
              </div>
            )}
            {inputs.min_age !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Min Age</p>
                <p className="text-white font-mono">{inputs.min_age} years</p>
              </div>
            )}
            {inputs.user_history !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">User History</p>
                <p className="text-white font-mono">{inputs.user_history}/100</p>
              </div>
            )}
            {inputs.limit_1h !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Limit 1h</p>
                <p className="text-white font-mono">${(inputs.limit_1h / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.limit_24h !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Limit 24h</p>
                <p className="text-white font-mono">${(inputs.limit_24h / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.daily_spent !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Daily Spent</p>
                <p className="text-white font-mono">${(inputs.daily_spent / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.daily_cap !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Daily Cap</p>
                <p className="text-white font-mono">${(inputs.daily_cap / 100).toFixed(2)}</p>
              </div>
            )}
            {inputs.min_trust !== undefined && (
              <div>
                <p className="text-gray-400 mb-1">Min Trust</p>
                <p className="text-white font-mono">{inputs.min_trust}/100</p>
              </div>
            )}
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
            <p className="text-xs text-gray-500 mt-3 italic">
              Note: A cryptographically valid proof can result in either approval or denial.
              The proof validates the computation was performed correctly.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
