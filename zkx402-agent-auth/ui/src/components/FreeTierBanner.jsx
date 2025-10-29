export default function FreeTierBanner() {
  return (
    <div className="bg-gradient-to-r from-accent-blue/10 to-accent-purple/10 border border-accent-blue/20 rounded-xl p-4 sm:p-6 mb-8 backdrop-blur-sm">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center gap-3 sm:gap-4">
          <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-accent-blue/20 to-accent-purple/20 rounded-full flex items-center justify-center border border-accent-blue/30">
            <svg className="w-5 h-5 sm:w-6 sm:h-6 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v13m0-13V6a2 2 0 112 2h-2zm0 0V5.5A2.5 2.5 0 109.5 8H12zm-7 4h14M5 12a2 2 0 110-4h14a2 2 0 110 4M5 12v7a2 2 0 002 2h10a2 2 0 002-2v-7" />
            </svg>
          </div>
          <div>
            <h3 className="text-base sm:text-lg font-semibold text-white mb-1">
              Free Test Access
            </h3>
            <p className="text-xs sm:text-sm text-gray-400">
              5 proof generations per day • No signup required
            </p>
          </div>
        </div>
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3">
          <a
            href="https://zk-x402.com/.well-known/x402"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs sm:text-sm px-4 py-2 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-gray-300 border border-dark-600 text-center"
          >
            View x402 Discovery
          </a>
          <a
            href="https://github.com/hshadab/zkx402/blob/main/zkx402-agent-auth/PAYMENT_GUIDE.md"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs sm:text-sm px-4 py-2 bg-gradient-to-r from-accent-green/20 to-accent-blue/20 hover:from-accent-green/30 hover:to-accent-blue/30 rounded-lg transition-colors text-accent-green border border-accent-green/30 font-medium text-center"
          >
            Production API →
          </a>
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-accent-blue/10 text-xs text-gray-400 flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-6">
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4 text-accent-green flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Unlimited production access via x402
        </span>
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4 text-accent-blue flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
          Pay per proof with Base USDC
        </span>
      </div>
    </div>
  );
}
