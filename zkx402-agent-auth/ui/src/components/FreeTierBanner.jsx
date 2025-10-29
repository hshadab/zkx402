export default function FreeTierBanner() {
  return (
    <div className="bg-gradient-to-r from-indigo-900/30 to-blue-900/30 border border-indigo-500/30 rounded-lg p-4 mb-8">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 text-2xl">
          🎁
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-accent-blue mb-2">
            Free Tier: 5 proof generations per day
          </h3>
          <div className="text-sm text-gray-300 space-y-2">
            <p>
              This UI is designed for <strong className="text-white">testing and demonstration</strong>.
              Generate up to 5 zkML proofs per day to explore our authorization models.
            </p>
            <div className="bg-dark-800/50 rounded p-3 mt-3">
              <p className="text-accent-green font-semibold mb-1">
                🚀 Production Use → x402 Protocol
              </p>
              <p className="text-gray-400">
                For unlimited pay-per-use access, integrate via the x402 protocol.
                Perfect for AI agents in production environments.
              </p>
              <div className="mt-2 flex gap-4 text-xs">
                <a
                  href="https://github.com/hshadab/zkx402"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent-blue hover:underline"
                >
                  📖 Documentation
                </a>
                <span className="text-gray-500">•</span>
                <span className="text-gray-400">
                  Endpoint: <code className="text-accent-green">/x402/authorize/:modelId</code>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
