import React, { useState } from 'react'
import Header from './components/Header'
import FreeTierBanner from './components/FreeTierBanner'
import ModelSelector from './components/ModelSelector'
import AuthorizationSimulator from './components/AuthorizationSimulator'
import ProofVisualization from './components/ProofVisualization'
import PerformanceMetrics from './components/PerformanceMetrics'
import ApiDocs from './components/ApiDocs'
import Analytics from './components/Analytics'
import HowToUse from './components/HowToUse'

function App() {
  const [selectedModel, setSelectedModel] = useState('percentage_limit')
  const [proofData, setProofData] = useState(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [viewMode, setViewMode] = useState('main') // 'howto' or 'main'
  const [activeTab, setActiveTab] = useState('simulator') // 'simulator', 'api', or 'analytics' (howto removed)

  // Handle hash-based navigation
  React.useEffect(() => {
    const handleHashChange = () => {
      if (window.location.hash === '#howto') {
        setViewMode('howto')
      } else if (window.location.hash === '#api') {
        setViewMode('main')
        setActiveTab('api')
      } else if (window.location.hash === '#simulator') {
        setViewMode('main')
        setActiveTab('simulator')
      } else if (window.location.hash === '#analytics') {
        setViewMode('main')
        setActiveTab('analytics')
      }
    }

    // Check on mount
    handleHashChange()

    // Listen for hash changes
    window.addEventListener('hashchange', handleHashChange)
    return () => window.removeEventListener('hashchange', handleHashChange)
  }, [])

  return (
    <div className="min-h-screen bg-dark-900">
      <Header />

      {/* Top-level Navigation */}
      <div className="bg-dark-800 border-b border-dark-700">
        <div className="container mx-auto px-4 py-3 max-w-7xl flex gap-4 items-center">
          <button
            onClick={() => setViewMode('howto')}
            className={`px-4 py-2 rounded-lg font-semibold transition-all text-sm sm:text-base ${
              viewMode === 'howto'
                ? 'bg-accent-green text-dark-900'
                : 'bg-dark-700 text-gray-300 hover:bg-dark-600 hover:text-white'
            }`}
          >
            📖 How to Use
          </button>
          <button
            onClick={() => setViewMode('main')}
            className={`px-4 py-2 rounded-lg font-semibold transition-all text-sm sm:text-base ${
              viewMode === 'main'
                ? 'bg-accent-green text-dark-900'
                : 'bg-dark-700 text-gray-300 hover:bg-dark-600 hover:text-white'
            }`}
          >
            🚀 zkX402 App
          </button>
        </div>
      </div>

      <main id="simulator" className="container mx-auto px-4 py-8 max-w-7xl">
        <FreeTierBanner />

        {/* How to Use Page - Separate top-level page */}
        {viewMode === 'howto' && (
          <div id="howto">
            <HowToUse />
          </div>
        )}

        {/* Main App - Contains the 3-tab interface */}
        {viewMode === 'main' && (
          <>
            {/* Tab Navigation - Only 3 tabs now */}
            <div className="flex gap-2 sm:gap-4 mb-8 border-b border-dark-700 overflow-x-auto">
              <button
                onClick={() => setActiveTab('simulator')}
                className={`px-4 sm:px-6 py-3 font-semibold transition-all whitespace-nowrap text-sm sm:text-base ${
                  activeTab === 'simulator'
                    ? 'text-accent-green border-b-2 border-accent-green'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                Verify Agent Models
              </button>
              <button
                onClick={() => setActiveTab('api')}
                className={`px-4 sm:px-6 py-3 font-semibold transition-all whitespace-nowrap text-sm sm:text-base ${
                  activeTab === 'api'
                    ? 'text-accent-green border-b-2 border-accent-green'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                API Documentation
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-4 sm:px-6 py-3 font-semibold transition-all whitespace-nowrap text-sm sm:text-base ${
                  activeTab === 'analytics'
                    ? 'text-accent-green border-b-2 border-accent-green'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                Analytics
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
              <div id="api">
                <ApiDocs />
              </div>
            )}

            {/* Analytics Tab */}
            {activeTab === 'analytics' && (
              <div id="analytics">
                <Analytics />
              </div>
            )}
          </>
        )}
      </main>

      <footer className="border-t border-dark-700 mt-8 sm:mt-16 py-6 sm:py-8">
        <div className="container mx-auto px-4 text-center text-gray-500 text-xs sm:text-sm">
          <p>Powered by JOLT Atlas zkML • Built for X402 Agent Authorization</p>
          <p className="mt-2">
            <a href="https://github.com/hshadab/zkx402" className="text-accent-green hover:underline">
              View on GitHub
            </a>
            <span className="mx-2">•</span>
            <button
              onClick={() => setActiveTab('analytics')}
              className="text-accent-green hover:underline cursor-pointer"
            >
              Analytics
            </button>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
