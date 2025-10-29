import React from 'react'

export default function Header() {
  return (
    <header className="bg-dark-800 border-b border-dark-700">
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-6">
            {/* NovaNet Logo */}
            <div className="flex items-center">
              <img
                src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
                alt="NovaNet"
                className="h-8 w-auto"
              />
            </div>

            {/* Service Info */}
            <div className="border-l border-dark-600 pl-6">
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-bold text-white">zkX402 Authorization</h1>
                <span className="px-2 py-1 text-xs font-semibold bg-accent-green/20 text-accent-green rounded">
                  x402 Infrastructure
                </span>
              </div>
              <p className="text-sm text-gray-400 mt-1">
                Zero-Knowledge ML Authorization for x402 Payment Agents
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* x402 Protocol Badge */}
            <a
              href="https://github.com/coinbase/x402"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-2 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors"
              title="Built on x402 Protocol"
            >
              <span className="text-sm font-medium text-gray-300">x402</span>
              <span className="text-xs text-gray-500">Protocol</span>
            </a>

            {/* GitHub Link */}
            <a
              href="https://github.com/hshadab/zkx402"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-accent-green transition-colors"
            >
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
            </a>
          </div>
        </div>

        {/* Infrastructure Notice */}
        <div className="mt-4 flex items-center gap-2 text-xs text-gray-500">
          <svg className="w-4 h-4 text-accent-green" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span>
            Powered by NovaNet • Core x402 Infrastructure Provider • Production Ready
          </span>
        </div>
      </div>
    </header>
  )
}
