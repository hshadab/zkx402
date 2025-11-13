/**
 * Application Constants
 * Centralized configuration values for zkX402
 */

module.exports = {
  // Cache configuration
  CACHE: {
    TTL_SECONDS: 86400,           // 24 hours
    DEFAULT_TTL: 86400
  },

  // Rate limiting
  RATE_LIMIT: {
    WINDOW_MS: 24 * 60 * 60 * 1000,   // 24 hours in milliseconds
    MAX_FREE_PROOFS: 5,                // Free proofs per IP per day
    MESSAGE: 'Free tier limit: 5 proofs per day. Use x402 payment for unlimited access.'
  },

  // File upload limits
  FILE_UPLOAD: {
    MAX_SIZE_BYTES: 100 * 1024 * 1024,  // 100 MB
    BUFFER_SIZE: 10 * 1024 * 1024       // 10 MB
  },

  // Payment verification
  PAYMENT: {
    MAX_TRANSACTION_AGE_SECONDS: 600,   // 10 minutes
    CONFIRMATION_BLOCKS: 1
  },

  // Webhook configuration
  WEBHOOK: {
    TIMEOUT_MS: 10000,           // 10 seconds
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY_MS: 2000         // 2 seconds between retries
  },

  // Analytics time windows
  ANALYTICS: {
    WINDOW_24H_MS: 24 * 60 * 60 * 1000,  // 24 hours
    WINDOW_7D_MS: 7 * 24 * 60 * 60 * 1000  // 7 days
  },

  // Blockchain monitoring
  BLOCKCHAIN: {
    CHUNK_SIZE: 10000,           // Blocks per chunk
    BLOCKS_24H: 43200,           // ~24 hours at 2s/block
    SCAN_DELAY_MS: 500           // Delay between chunks
  }
};
