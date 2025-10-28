import React, { useState } from 'react'
import Header from './components/Header'
import ModelSelector from './components/ModelSelector'
import AuthorizationSimulator from './components/AuthorizationSimulator'
import ProofVisualization from './components/ProofVisualization'
import PerformanceMetrics from './components/PerformanceMetrics'
import ApiDocs from './components/ApiDocs'

function App() {
  const [selectedModel, setSelectedModel] = useState('simple_threshold')
  const [proofData, setProofData] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [activeTab, setActiveTab] = useState('simulator') // 'simulator' or 'api'

  return (
    <div className="min-h-screen bg-dark-900">
      <Header />

      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Tab Navigation */}
        <div className="flex gap-4 mb-8 border-b border-dark-700">
          <button
            onClick={() => setActiveTab('simulator')}
            className={`px-6 py-3 font-semibold transition-all ${
              activeTab === 'simulator'
                ? 'text-accent-green border-b-2 border-accent-green'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Authorization Simulator
          </button>
          <button
            onClick={() => setActiveTab('api')}
            className={`px-6 py-3 font-semibold transition-all ${
              activeTab === 'api'
                ? 'text-accent-green border-b-2 border-accent-green'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            API Documentation
          </button>
        </div>

        {/* Simulator Tab */}
        {activeTab === 'simulator' && (
          <>
            {/* Model Selection */}
            <section className="mb-8">
              <ModelSelector
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
              />
            </section>

            {/* Authorization Simulator */}
            <section className="mb-8">
              <AuthorizationSimulator
                modelType={selectedModel}
                onProofGenerated={setProofData}
                isGenerating={isGenerating}
                setIsGenerating={setIsGenerating}
              />
            </section>

            {/* Proof Visualization */}
            {proofData && (
              <>
                <section className="mb-8">
                  <ProofVisualization proofData={proofData} />
                </section>

                <section className="mb-8">
                  <PerformanceMetrics proofData={proofData} />
                </section>
              </>
            )}

            {/* Info Section */}
            <section className="card mt-12">
              <h2 className="card-header">About zkX402</h2>
              <div className="space-y-4 text-gray-300">
                <p>
                  zkX402 enables AI agents to prove authorization without revealing private data
                  using zero-knowledge machine learning proofs powered by JOLT Atlas.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                  <div className="bg-dark-700 p-4 rounded-lg">
                    <h3 className="text-accent-green font-semibold mb-2">Rule-Based</h3>
                    <p className="text-sm">Simple threshold checks and comparisons</p>
                  </div>
                  <div className="bg-dark-700 p-4 rounded-lg">
                    <h3 className="text-accent-blue font-semibold mb-2">Neural Network</h3>
                    <p className="text-sm">ML-based risk scoring and classification</p>
                  </div>
                  <div className="bg-dark-700 p-4 rounded-lg">
                    <h3 className="text-accent-purple font-semibold mb-2">Hybrid</h3>
                    <p className="text-sm">Combined rule-based and ML authorization</p>
                  </div>
                </div>
              </div>
            </section>
          </>
        )}

        {/* API Docs Tab */}
        {activeTab === 'api' && (
          <ApiDocs />
        )}
      </main>

      <footer className="border-t border-dark-700 mt-16 py-8">
        <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
          <p>Powered by JOLT Atlas zkML • Built for X402 Agent Authorization</p>
          <p className="mt-2">
            <a href="https://github.com/hshadab/zkx402" className="text-accent-green hover:underline">
              View on GitHub
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
