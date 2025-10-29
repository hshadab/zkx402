import React from 'react'

export default function Header() {
  return (
    <>
      {/* Top Navigation Bar */}
      <nav className="bg-dark-900/80 backdrop-blur-sm border-b border-dark-700 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 max-w-7xl">
          <div className="flex items-center justify-between">
            {/* Logo and Brand */}
            <div className="flex items-center gap-4">
              <a
                href="https://www.novanet.xyz/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center"
              >
                <img
                  src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
                  alt="NovaNet"
                  className="h-7 w-auto opacity-90 hover:opacity-100 transition-opacity"
                />
              </a>
              <div className="h-6 w-px bg-dark-600"></div>
              <span className="text-xl font-bold bg-gradient-to-r from-accent-green to-accent-blue bg-clip-text text-transparent">
                zkx402
              </span>
            </div>

            {/* Navigation Links */}
            <div className="flex items-center gap-6">
              <a
                href="#features"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Features
              </a>
              <a
                href="https://www.novanet.xyz/blog/x402-zkml-internet-native-verifiable-payments"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                About NovaNet
              </a>
              <a
                href="https://github.com/hshadab/zkx402"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                Documentation
              </a>
              <a
                href="https://github.com/coinbase/x402"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm px-3 py-1.5 bg-dark-700 hover:bg-dark-600 rounded-md transition-colors text-gray-300"
              >
                x402 Protocol
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="relative bg-gradient-to-br from-dark-900 via-dark-800 to-dark-900 border-b border-dark-700 overflow-hidden">
        {/* Animated Background Grid */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: 'linear-gradient(rgba(52, 211, 153, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(52, 211, 153, 0.1) 1px, transparent 1px)',
            backgroundSize: '50px 50px'
          }}></div>
        </div>

        {/* Gradient Orbs */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent-green/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent-blue/10 rounded-full blur-3xl"></div>

        <div className="container mx-auto px-4 py-16 max-w-7xl relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-dark-700/50 backdrop-blur-sm border border-accent-green/20 rounded-full mb-8">
              <svg className="w-4 h-4 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <span className="text-sm font-medium text-gray-300">Powered by JOLT Atlas zkML</span>
            </div>

            {/* Main Heading */}
            <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
                Verifiable
              </span>
              <br />
              <span className="bg-gradient-to-r from-accent-green via-accent-blue to-accent-purple bg-clip-text text-transparent">
                x402 Agents
              </span>
            </h1>

            {/* Tagline */}
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto leading-relaxed">
              Zero-knowledge machine learning proofs for the x402 payment protocol.
              Prove authorization decisions without revealing private data.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-4 mb-12">
              <a
                href="#simulator"
                className="px-8 py-3 bg-gradient-to-r from-accent-green to-accent-blue hover:from-accent-green/90 hover:to-accent-blue/90 text-dark-900 font-semibold rounded-lg transition-all shadow-lg shadow-accent-green/20"
              >
                Try zkML
              </a>
              <a
                href="#api"
                className="px-8 py-3 bg-dark-700 hover:bg-dark-600 text-white font-semibold rounded-lg transition-colors border border-dark-600"
              >
                View API
              </a>
            </div>

            {/* Feature Pills */}
            <div id="features" className="flex flex-wrap items-center justify-center gap-3 text-sm">
              <div className="flex items-center gap-2 px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-4 h-4 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="text-gray-300">~700ms Proofs</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-4 h-4 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span className="text-gray-300">Privacy-Preserving</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-4 h-4 text-accent-purple" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-gray-300">14 Models</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-4 h-4 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                <span className="text-gray-300">Base USDC Payments</span>
              </div>
            </div>
          </div>
        </div>
      </header>
    </>
  )
}
