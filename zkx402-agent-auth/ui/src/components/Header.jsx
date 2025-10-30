import React, { useState } from 'react'

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <>
      {/* Top Navigation Bar */}
      <nav className="bg-dark-900/80 backdrop-blur-sm border-b border-dark-700 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 max-w-7xl">
          <div className="flex items-center justify-between">
            {/* Logo and Brand */}
            <div className="flex items-center gap-2 sm:gap-4">
              <a
                href="/"
                onClick={(e) => {
                  e.preventDefault()
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                }}
                className="flex items-center cursor-pointer"
              >
                <img
                  src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
                  alt="NovaNet"
                  className="h-6 sm:h-7 w-auto opacity-90 hover:opacity-100 transition-opacity"
                />
              </a>
              <div className="h-6 w-px bg-dark-600 hidden sm:block"></div>
              <a
                href="/"
                onClick={(e) => {
                  e.preventDefault()
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                }}
                className="cursor-pointer"
              >
                <span className="text-lg sm:text-xl font-bold bg-gradient-to-r from-accent-green to-accent-blue bg-clip-text text-transparent">
                  zkx402
                </span>
              </a>
            </div>

            {/* Desktop Navigation Links */}
            <div className="hidden lg:flex items-center gap-6">
              <a
                href="#howto"
                className="text-sm text-gray-400 hover:text-white transition-colors"
              >
                How to Use
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

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 text-gray-400 hover:text-white transition-colors"
              aria-label="Toggle menu"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>

          {/* Mobile Navigation Menu */}
          {mobileMenuOpen && (
            <div className="lg:hidden mt-4 py-4 border-t border-dark-700">
              <div className="flex flex-col gap-4">
                <a
                  href="#howto"
                  onClick={() => setMobileMenuOpen(false)}
                  className="text-sm text-gray-400 hover:text-white transition-colors py-2"
                >
                  How to Use
                </a>
                <a
                  href="https://www.novanet.xyz/blog/x402-zkml-internet-native-verifiable-payments"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-gray-400 hover:text-white transition-colors py-2"
                >
                  About NovaNet
                </a>
                <a
                  href="https://github.com/hshadab/zkx402"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-gray-400 hover:text-white transition-colors py-2"
                >
                  Documentation
                </a>
                <a
                  href="https://github.com/coinbase/x402"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm px-3 py-2 bg-dark-700 hover:bg-dark-600 rounded-md transition-colors text-gray-300 text-center"
                >
                  x402 Protocol
                </a>
              </div>
            </div>
          )}
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

        <div className="container mx-auto px-4 py-8 sm:py-12 md:py-16 max-w-7xl relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            {/* Badge */}
            <a
              href="https://github.com/ICME-Lab/jolt-atlas"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-3 sm:px-4 py-2 bg-dark-700/50 backdrop-blur-sm border border-accent-green/20 rounded-full mb-6 sm:mb-8 hover:border-accent-green/40 transition-colors"
            >
              <svg className="w-4 h-4 text-accent-green flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <span className="text-xs sm:text-sm font-medium text-gray-300">Powered by JOLT Atlas zkML</span>
            </a>

            {/* Main Heading */}
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold mb-4 sm:mb-6 leading-tight px-4">
              <span className="bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
                Verifiable
              </span>
              <br />
              <span className="bg-gradient-to-r from-accent-green via-accent-blue to-accent-purple bg-clip-text text-transparent">
                x402 Agents
              </span>
            </h1>

            {/* Tagline */}
            <p className="text-base sm:text-lg md:text-xl text-gray-400 mb-6 sm:mb-8 max-w-2xl mx-auto leading-relaxed px-4">
              Zero-knowledge machine learning proofs for the x402 payment protocol.
              Prove authorization decisions without revealing private data.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 mb-8 sm:mb-12 px-4">
              <a
                href="#simulator"
                className="w-full sm:w-auto px-6 sm:px-8 py-3 bg-gradient-to-r from-accent-green to-accent-blue hover:from-accent-green/90 hover:to-accent-blue/90 text-dark-900 font-semibold rounded-lg transition-all shadow-lg shadow-accent-green/20 text-center"
              >
                Try zkML
              </a>
              <a
                href="#api"
                className="w-full sm:w-auto px-6 sm:px-8 py-3 bg-dark-700 hover:bg-dark-600 text-white font-semibold rounded-lg transition-colors border border-dark-600 text-center"
              >
                View API
              </a>
            </div>

            {/* Feature Pills */}
            <div id="features" className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 text-xs sm:text-sm px-4">
              <div className="flex items-center gap-2 px-3 sm:px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-3 h-3 sm:w-4 sm:h-4 text-accent-green flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="text-gray-300 whitespace-nowrap">~700ms Proofs</span>
              </div>
              <div className="flex items-center gap-2 px-3 sm:px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-3 h-3 sm:w-4 sm:h-4 text-accent-blue flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span className="text-gray-300 whitespace-nowrap">Privacy-Preserving</span>
              </div>
              <div className="flex items-center gap-2 px-3 sm:px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-3 h-3 sm:w-4 sm:h-4 text-accent-purple flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-gray-300 whitespace-nowrap">14 Models</span>
              </div>
              <div className="flex items-center gap-2 px-3 sm:px-4 py-2 bg-dark-700/50 rounded-full border border-dark-600">
                <svg className="w-3 h-3 sm:w-4 sm:h-4 text-accent-green flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
                <span className="text-gray-300 whitespace-nowrap">Base USDC</span>
              </div>
            </div>
          </div>
        </div>
      </header>
    </>
  )
}
