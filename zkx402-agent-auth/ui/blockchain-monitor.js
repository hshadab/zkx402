/**
 * Blockchain Payment Monitor for zkX402
 * Fetches real USDC transactions from Base network
 */

const { ethers } = require('ethers');

// Base Mainnet Configuration
const BASE_RPC_URL = 'https://mainnet.base.org';
const USDC_CONTRACT = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';
const PAYMENT_WALLET = '0x1f409E94684804e5158561090Ced8941B47B0CC6';

// USDC ABI - Transfer event only
const USDC_ABI = [
  'event Transfer(address indexed from, address indexed to, uint256 value)',
  'function balanceOf(address account) view returns (uint256)',
  'function decimals() view returns (uint8)',
];

class BlockchainMonitor {
  constructor() {
    this.provider = new ethers.JsonRpcProvider(BASE_RPC_URL);
    this.usdcContract = new ethers.Contract(USDC_CONTRACT, USDC_ABI, this.provider);
    this.cachedTransactions = [];
    this.lastFetchedBlock = null;
    this.isInitialized = false;
  }

  /**
   * Initialize the monitor and fetch historical transactions
   */
  async initialize() {
    try {
      console.log('🔗 Initializing blockchain monitor for Base network...');

      // Get current block
      const currentBlock = await this.provider.getBlockNumber();
      console.log(`📦 Current Base block: ${currentBlock}`);

      // Fetch transactions from last 24 hours (~43,200 blocks at 2s/block)
      // Reduced from 7 days to avoid RPC rate limits
      const blocksToScan = 43200; // ~24 hours
      const fromBlock = Math.max(0, currentBlock - blocksToScan);

      await this.fetchTransactionsChunked(fromBlock, currentBlock);

      this.lastFetchedBlock = currentBlock;
      this.isInitialized = true;

      console.log(`✅ Blockchain monitor initialized. Found ${this.cachedTransactions.length} transactions.`);
    } catch (error) {
      console.error('❌ Failed to initialize blockchain monitor:', error.message);
    }
  }

  /**
   * Fetch transactions in chunks to avoid RPC rate limits
   */
  async fetchTransactionsChunked(fromBlock, toBlock) {
    const CHUNK_SIZE = 10000; // Process 10k blocks at a time (~5.5 hours)
    let currentFrom = fromBlock;

    console.log(`🔍 Scanning blocks ${fromBlock} to ${toBlock} in chunks of ${CHUNK_SIZE}...`);

    while (currentFrom < toBlock) {
      const currentTo = Math.min(currentFrom + CHUNK_SIZE, toBlock);
      await this.fetchTransactions(currentFrom, currentTo);

      // Small delay to avoid rate limiting
      if (currentTo < toBlock) {
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      currentFrom = currentTo + 1;
    }

    console.log(`✅ Finished scanning. Found ${this.cachedTransactions.length} total transactions`);
  }

  /**
   * Fetch USDC transfer events to our wallet
   */
  async fetchTransactions(fromBlock, toBlock) {
    try {
      console.log(`  📦 Chunk: blocks ${fromBlock} to ${toBlock}`);

      // Query Transfer events where 'to' is our payment wallet
      const filter = this.usdcContract.filters.Transfer(null, PAYMENT_WALLET);
      const events = await this.usdcContract.queryFilter(filter, fromBlock, toBlock);

      if (events.length > 0) {
        console.log(`  📋 Found ${events.length} transfer events in this chunk`);
      }

      for (const event of events) {
        const block = await event.getBlock();
        const tx = await event.getTransaction();

        const transaction = {
          id: `blockchain_${event.transactionHash}`,
          txHash: event.transactionHash,
          from: event.args.from,
          to: event.args.to,
          amount: event.args.value.toString(),
          amountUSDC: ethers.formatUnits(event.args.value, 6), // USDC has 6 decimals
          blockNumber: event.blockNumber,
          timestamp: new Date(block.timestamp * 1000).toISOString(),
          verified: true,
          source: 'blockchain',
          gasUsed: tx.gasLimit ? tx.gasLimit.toString() : null,
        };

        // Check if we already have this transaction
        const exists = this.cachedTransactions.find(t => t.txHash === transaction.txHash);
        if (!exists) {
          this.cachedTransactions.push(transaction);
        }
      }

      // Sort by timestamp (newest first)
      this.cachedTransactions.sort((a, b) =>
        new Date(b.timestamp) - new Date(a.timestamp)
      );
    } catch (error) {
      console.error(`  ❌ Error fetching chunk ${fromBlock}-${toBlock}:`, error.message);
      // Continue even if one chunk fails
    }
  }

  /**
   * Refresh transactions (fetch new ones since last check)
   */
  async refresh() {
    if (!this.isInitialized) {
      await this.initialize();
      return;
    }

    try {
      const currentBlock = await this.provider.getBlockNumber();

      if (currentBlock > this.lastFetchedBlock) {
        await this.fetchTransactions(this.lastFetchedBlock + 1, currentBlock);
        this.lastFetchedBlock = currentBlock;
      }
    } catch (error) {
      console.error('❌ Error refreshing transactions:', error.message);
    }
  }

  /**
   * Get all cached blockchain transactions
   */
  getTransactions() {
    return this.cachedTransactions;
  }

  /**
   * Get blockchain payment statistics
   */
  async getStats() {
    const now = new Date();
    const last24h = new Date(now - 24 * 60 * 60 * 1000);
    const last7d = new Date(now - 7 * 24 * 60 * 60 * 1000);

    const payments24h = this.cachedTransactions.filter(
      t => new Date(t.timestamp) > last24h
    );
    const payments7d = this.cachedTransactions.filter(
      t => new Date(t.timestamp) > last7d
    );

    const totalRevenue = this.cachedTransactions.reduce(
      (sum, t) => sum + parseFloat(t.amountUSDC), 0
    );
    const revenue24h = payments24h.reduce(
      (sum, t) => sum + parseFloat(t.amountUSDC), 0
    );
    const revenue7d = payments7d.reduce(
      (sum, t) => sum + parseFloat(t.amountUSDC), 0
    );

    // Get current wallet balance
    let currentBalance = '0';
    try {
      const balance = await this.usdcContract.balanceOf(PAYMENT_WALLET);
      currentBalance = ethers.formatUnits(balance, 6);
    } catch (error) {
      console.error('Failed to fetch wallet balance:', error.message);
    }

    return {
      totalTransactions: this.cachedTransactions.length,
      transactions24h: payments24h.length,
      transactions7d: payments7d.length,
      totalRevenue: totalRevenue.toFixed(4),
      revenue24h: revenue24h.toFixed(4),
      revenue7d: revenue7d.toFixed(4),
      currentBalance,
      recentTransactions: this.cachedTransactions.slice(0, 10),
    };
  }

  /**
   * Get a specific transaction by hash
   */
  getTransaction(txHash) {
    return this.cachedTransactions.find(t => t.txHash === txHash);
  }
}

// Controlled initialization via env flag (default ON for compatibility)
const ENABLED = process.env.ENABLE_BLOCKCHAIN_MONITOR === '0' ? false : true;

if (!ENABLED) {
  module.exports = {
    async getStats() {
      return { enabled: false };
    }
  };
} else {
  // Export singleton instance
  const monitor = new BlockchainMonitor();
  // Auto-initialize on first require
  monitor.initialize();
  // Auto-refresh every 2 minutes
  setInterval(() => {
    monitor.refresh();
  }, 2 * 60 * 1000);
  module.exports = monitor;
}
