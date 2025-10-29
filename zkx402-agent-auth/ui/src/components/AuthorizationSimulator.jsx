import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { getModel, getInputConfig, formatInputValue } from '../utils/curatedModels'

export default function AuthorizationSimulator({ modelType, onProofGenerated, isGenerating, setIsGenerating }) {
  const model = getModel(modelType)
  const [inputs, setInputs] = useState({})
  const [showExamples, setShowExamples] = useState(false)

  // Initialize inputs with default values when model changes
  useEffect(() => {
    if (model && model.examples && model.examples.length > 0) {
      // Use first example as default
      const defaultInputs = {}
      model.inputs.forEach(inputName => {
        defaultInputs[inputName] = model.examples[0][inputName] || ''
      })
      setInputs(defaultInputs)
    }
  }, [model])

  const handleInputChange = (field, value) => {
    setInputs(prev => ({ ...prev, [field]: value }))
  }

  const loadExample = (example) => {
    const newInputs = {}
    model.inputs.forEach(inputName => {
      newInputs[inputName] = example[inputName] || ''
    })
    setInputs(newInputs)
    setShowExamples(false)
  }

  const generateProof = async () => {
    setIsGenerating(true)
    try {
      const startTime = Date.now()

      // Call backend API
      const response = await axios.post('/api/generate-proof', {
        model: modelType,
        inputs: inputs
      })

      const proofTime = Date.now() - startTime

      onProofGenerated({
        ...response.data,
        proofTime,
        modelType,
        inputs
      })
    } catch (error) {
      console.error('Proof generation failed:', error)
      alert('Proof generation failed: ' + (error.response?.data?.error || error.message))
    } finally {
      setIsGenerating(false)
    }
  }

  if (!model) {
    return (
      <div className="card">
        <h2 className="card-header">Agent zkML Generator</h2>
        <div className="text-center py-12 text-gray-500">
          <p>Please select a model to begin</p>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold">Agent zkML Generator</h2>
            <p className="text-sm text-gray-400 mt-1">
              Test {model.name} with custom inputs
            </p>
          </div>
          {model.examples && model.examples.length > 0 && (
            <button
              onClick={() => setShowExamples(!showExamples)}
              className="btn-secondary text-sm"
            >
              {showExamples ? 'Hide' : 'Show'} Examples
            </button>
          )}
        </div>
      </div>

      {/* Example Scenarios */}
      {showExamples && model.examples && (
        <div className="mb-6 p-4 bg-dark-700 rounded-lg border border-dark-600">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Example Scenarios</h3>
          <div className="space-y-2">
            {model.examples.map((example, idx) => (
              <button
                key={idx}
                onClick={() => loadExample(example)}
                className="w-full p-3 text-left bg-dark-600 hover:bg-dark-500 rounded-lg transition-all"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm font-medium text-white">{example.desc}</span>
                    <div className="flex gap-3 mt-1 text-xs text-gray-400">
                      {model.inputs.map(inputName => (
                        <span key={inputName}>
                          {inputName}: {formatInputValue(inputName, example[inputName])}
                        </span>
                      ))}
                    </div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    example.expected === 'approved'
                      ? 'bg-accent-green/20 text-accent-green'
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {example.expected === 'approved' ? '✓ Approved' : '✗ Denied'}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-4">
        <p className="text-gray-400 text-sm mb-6">
          {model.description} - {model.useCase}
        </p>

        {/* Dynamic Input Fields */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {model.inputs.map(inputName => {
            const config = getInputConfig(inputName)
            return (
              <div key={inputName}>
                <label className="block text-sm font-medium mb-2 text-gray-300">
                  {config.label}
                  {config.example && (
                    <span className="text-gray-500 ml-2 font-normal">({config.example})</span>
                  )}
                </label>
                <input
                  type={config.type}
                  value={inputs[inputName] || ''}
                  onChange={(e) => handleInputChange(inputName, e.target.value)}
                  className="input-field w-full"
                  placeholder={config.placeholder}
                  min={config.min}
                  max={config.max}
                />
                {config.description && (
                  <p className="text-xs text-gray-500 mt-1">{config.description}</p>
                )}
              </div>
            )
          })}
        </div>

        {/* Current Values Display */}
        <div className="mt-6 p-4 bg-dark-700 rounded-lg border border-dark-600">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Current Values</h3>
          <div className="flex flex-wrap gap-3">
            {model.inputs.map(inputName => (
              <div key={inputName} className="bg-dark-600 px-3 py-2 rounded">
                <div className="text-xs text-gray-500">{getInputConfig(inputName).label}</div>
                <div className="text-sm font-medium text-white">
                  {inputs[inputName] ? formatInputValue(inputName, inputs[inputName]) : '—'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Generate Proof Button */}
        <div className="pt-4">
          <button
            onClick={generateProof}
            disabled={isGenerating || !model.inputs.every(input => inputs[input])}
            className="btn-primary w-full md:w-auto"
          >
            {isGenerating ? (
              <>
                <span className="inline-block animate-spin mr-2">⚙</span>
                Generating zkML Proof...
              </>
            ) : (
              'Generate Zero-Knowledge Proof'
            )}
          </button>
          {!model.inputs.every(input => inputs[input]) && (
            <p className="text-xs text-gray-500 mt-2">
              Please fill in all input fields
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
