import React, { useState } from 'react'
import axios from 'axios'

export default function AuthorizationSimulator({ modelType, onProofGenerated, isGenerating, setIsGenerating }) {
  const [inputs, setInputs] = useState({
    amount: '500',
    balance: '10000',
    velocity_1h: '200',
    velocity_24h: '1500',
    vendor_trust: '80'
  })

  const handleInputChange = (field, value) => {
    setInputs(prev => ({ ...prev, [field]: value }))
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

  return (
    <div className="card">
      <h2 className="card-header">Authorization Simulator</h2>

      <div className="space-y-4">
        <p className="text-gray-400 text-sm mb-6">
          Simulate agent authorization with private financial data. All values scaled by 100 (e.g., 500 = $5.00)
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              Transaction Amount
              <span className="text-gray-500 ml-2">(scaled, e.g., 500 = $5.00)</span>
            </label>
            <input
              type="number"
              value={inputs.amount}
              onChange={(e) => handleInputChange('amount', e.target.value)}
              className="input-field w-full"
              placeholder="500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              Account Balance
              <span className="text-gray-500 ml-2">(scaled, e.g., 10000 = $100.00)</span>
            </label>
            <input
              type="number"
              value={inputs.balance}
              onChange={(e) => handleInputChange('balance', e.target.value)}
              className="input-field w-full"
              placeholder="10000"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              1-Hour Velocity
              <span className="text-gray-500 ml-2">(scaled spending rate)</span>
            </label>
            <input
              type="number"
              value={inputs.velocity_1h}
              onChange={(e) => handleInputChange('velocity_1h', e.target.value)}
              className="input-field w-full"
              placeholder="200"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              24-Hour Velocity
              <span className="text-gray-500 ml-2">(scaled spending rate)</span>
            </label>
            <input
              type="number"
              value={inputs.velocity_24h}
              onChange={(e) => handleInputChange('velocity_24h', e.target.value)}
              className="input-field w-full"
              placeholder="1500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-300">
              Vendor Trust Score
              <span className="text-gray-500 ml-2">(0-100)</span>
            </label>
            <input
              type="number"
              value={inputs.vendor_trust}
              onChange={(e) => handleInputChange('vendor_trust', e.target.value)}
              className="input-field w-full"
              placeholder="80"
              min="0"
              max="100"
            />
          </div>
        </div>

        <div className="pt-4">
          <button
            onClick={generateProof}
            disabled={isGenerating}
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
        </div>
      </div>
    </div>
  )
}
