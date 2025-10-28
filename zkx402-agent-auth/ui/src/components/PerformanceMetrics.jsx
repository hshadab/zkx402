import React from 'react'

export default function PerformanceMetrics({ proofData }) {
  const { proofTime, proofSize = 'N/A', verificationTime = 'N/A' } = proofData

  const metrics = [
    {
      label: 'Proof Generation',
      value: `${(proofTime / 1000).toFixed(2)}s`,
      description: 'Time to generate zkML proof',
      color: 'accent-green'
    },
    {
      label: 'Proof Size',
      value: proofSize,
      description: 'Compressed proof size',
      color: 'accent-blue'
    },
    {
      label: 'Verification Time',
      value: verificationTime,
      description: 'Time to verify proof',
      color: 'accent-purple'
    },
    {
      label: 'Operations',
      value: proofData.operations || 'N/A',
      description: 'ONNX operations executed',
      color: 'accent-green'
    }
  ]

  return (
    <div className="card">
      <h2 className="card-header">Performance Metrics</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, index) => (
          <div
            key={index}
            className="bg-dark-700 p-4 rounded-lg border border-dark-600"
          >
            <p className="text-gray-400 text-sm mb-2">{metric.label}</p>
            <p className={`text-2xl font-bold text-${metric.color} mb-1`}>
              {metric.value}
            </p>
            <p className="text-gray-500 text-xs">{metric.description}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-dark-700 rounded-lg border border-dark-600">
        <h3 className="text-sm font-semibold mb-3 text-gray-300">System Benchmarks</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <p className="text-gray-400 mb-1">Simple Rules</p>
            <p className="text-white">~0.5s • 15 KB proof</p>
          </div>
          <div>
            <p className="text-gray-400 mb-1">Neural Network</p>
            <p className="text-white">~1.5s • 40 KB proof</p>
          </div>
          <div>
            <p className="text-gray-400 mb-1">Complex NN</p>
            <p className="text-white">~3.0s • 80 KB proof</p>
          </div>
        </div>
      </div>
    </div>
  )
}
