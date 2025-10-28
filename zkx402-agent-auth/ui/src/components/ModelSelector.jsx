import React from 'react'

const MODELS = [
  {
    id: 'simple_auth',
    name: 'Simple Rule-Based',
    description: 'Threshold checks and comparison operations',
    operations: '10-20 ops',
    proofTime: '~0.5s',
    color: 'accent-green'
  },
  {
    id: 'neural_auth',
    name: 'Neural Network',
    description: 'ML-based risk scoring (8→4→1 architecture)',
    operations: '20-50 ops',
    proofTime: '~1.5s',
    color: 'accent-blue'
  },
  {
    id: 'comparison_demo',
    name: 'Comparison Demo',
    description: 'Greater, Less, GreaterEqual operations',
    operations: '5-10 ops',
    proofTime: '~0.3s',
    color: 'accent-purple'
  },
  {
    id: 'tensor_ops_demo',
    name: 'Tensor Operations',
    description: 'Slice, Identity, Reshape demo',
    operations: '5-10 ops',
    proofTime: '~0.3s',
    color: 'accent-green'
  },
  {
    id: 'matmult_1d_demo',
    name: 'MatMult 1D',
    description: 'Matrix multiplication with 1D outputs',
    operations: '5-10 ops',
    proofTime: '~0.4s',
    color: 'accent-blue'
  }
]

export default function ModelSelector({ selectedModel, onModelChange }) {
  return (
    <div className="card">
      <h2 className="card-header">Select Authorization Model</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {MODELS.map(model => (
          <button
            key={model.id}
            onClick={() => onModelChange(model.id)}
            className={`
              p-4 rounded-lg border-2 text-left transition-all
              ${selectedModel === model.id
                ? `border-${model.color} bg-${model.color}/10`
                : 'border-dark-600 bg-dark-700 hover:border-dark-500'
              }
            `}
          >
            <h3 className={`font-semibold mb-2 ${selectedModel === model.id ? `text-${model.color}` : 'text-white'}`}>
              {model.name}
            </h3>
            <p className="text-sm text-gray-400 mb-3">{model.description}</p>
            <div className="flex gap-4 text-xs text-gray-500">
              <span>{model.operations}</span>
              <span>•</span>
              <span>{model.proofTime}</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}
