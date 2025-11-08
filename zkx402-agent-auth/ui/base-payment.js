/**
 * Base Network Payment Verification for zkX402
 *
 * Verifies USDC payments on Base L2 for x402 protocol
 * Network: Base Mainnet (Chain ID: 8453)
 * Payment Token: USDC (6 decimals)
 */

const { ethers } = require('ethers');

// Base Network Configuration
const BASE_MAINNET = {
  chainId: 8453,
  name: 'Base Mainnet',
  rpcUrl: 'https://mainnet.base.org',
  explorer: 'https://basescan.org'
};

// USDC Contract on Base Mainnet
const USDC_BASE = {
  address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
  decimals: 6,
  symbol: 'USDC'
};

// ERC20 ABI (minimal - just what we need)
const ERC20_ABI = [
  'function transfer(address to, uint256 amount) returns (bool)',
  'function balanceOf(address account) view returns (uint256)',
  'event Transfer(address indexed from, address indexed to, uint256 value)'
];

// Your zkX402 payment wallet (configurable via environment variable)
const PAYMENT_WALLET = process.env.PAYMENT_WALLET || '0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91';

/**
 * Initialize Base provider
 */
function getBaseProvider() {
  return new ethers.JsonRpcProvider(BASE_MAINNET.rpcUrl);
}

/**
 * Verify a USDC payment transaction on Base
 *
 * @param {string} txHash - Transaction hash
 * @param {string} expectedAmount - Expected amount in USDC atomic units (e.g., "1000" = 0.001 USDC)
 * @param {number} maxAgeSeconds - Maximum age of transaction in seconds (default: 600 = 10 minutes)
 * @returns {Promise<{isValid: boolean, invalidReason?: string, details?: object}>}
 */
async function verifyPayment(txHash, expectedAmount, maxAgeSeconds = 600) {
  try {
    const provider = getBaseProvider();

    // Get transaction receipt
    const receipt = await provider.getTransactionReceipt(txHash);

    if (!receipt) {
      return {
        isValid: false,
        invalidReason: 'Transaction not found or not yet mined'
      };
    }

    // Check transaction is confirmed (at least 1 block)
    if (!receipt.status) {
      return {
        isValid: false,
        invalidReason: 'Transaction failed on-chain'
      };
    }

    // Get current block to check transaction age
    const currentBlock = await provider.getBlockNumber();
    const txBlock = await provider.getBlock(receipt.blockNumber);
    const txAge = Date.now() / 1000 - txBlock.timestamp;

    if (txAge > maxAgeSeconds) {
      return {
        isValid: false,
        invalidReason: `Transaction too old (${Math.floor(txAge)}s > ${maxAgeSeconds}s max)`
      };
    }

    // Parse Transfer events from USDC contract
    const usdcInterface = new ethers.Interface(ERC20_ABI);
    let paymentFound = false;
    let actualAmount = '0';
    let sender = null;

    for (const log of receipt.logs) {
      // Check if this is a USDC transfer event
      if (log.address.toLowerCase() !== USDC_BASE.address.toLowerCase()) {
        continue;
      }

      try {
        const parsedLog = usdcInterface.parseLog(log);

        if (parsedLog.name === 'Transfer') {
          const to = parsedLog.args[1];
          const amount = parsedLog.args[2];

          // Check if transfer was to our payment wallet
          if (to.toLowerCase() === PAYMENT_WALLET.toLowerCase()) {
            paymentFound = true;
            actualAmount = amount.toString();
            sender = parsedLog.args[0];
            break;
          }
        }
      } catch (e) {
        // Not a Transfer event or parsing failed, continue
        continue;
      }
    }

    if (!paymentFound) {
      return {
        isValid: false,
        invalidReason: `No USDC transfer found to payment wallet ${PAYMENT_WALLET}`
      };
    }

    // Verify amount matches (USDC has 6 decimals)
    // expectedAmount is in our atomic units (1000 = $0.01 USDC)
    // We need to convert to USDC atomic units (1 USDC = 1,000,000 units)
    // Conversion: 1000 atomic units = $0.01 = 10,000 USDC units
    const expectedUsdcUnits = BigInt(expectedAmount) * BigInt(10); // Convert our units to USDC's 6-decimal units
    const actualUsdcUnits = BigInt(actualAmount);

    if (actualUsdcUnits < expectedUsdcUnits) {
      return {
        isValid: false,
        invalidReason: `Insufficient payment: received ${actualAmount} USDC units, expected ${expectedUsdcUnits} USDC units`,
        details: {
          received: actualAmount,
          expected: expectedUsdcUnits.toString(),
          receivedUSDC: (Number(actualAmount) / 1_000_000).toFixed(6),
          expectedUSDC: (Number(expectedUsdcUnits) / 1_000_000).toFixed(6)
        }
      };
    }

    // Payment verified!
    return {
      isValid: true,
      details: {
        txHash,
        sender,
        recipient: PAYMENT_WALLET,
        amount: actualAmount,
        amountUSDC: (Number(actualAmount) / 1_000_000).toFixed(6),
        blockNumber: receipt.blockNumber,
        confirmations: currentBlock - receipt.blockNumber,
        timestamp: txBlock.timestamp,
        age: Math.floor(txAge),
        explorer: `${BASE_MAINNET.explorer}/tx/${txHash}`
      }
    };

  } catch (error) {
    return {
      isValid: false,
      invalidReason: `Payment verification error: ${error.message}`
    };
  }
}

/**
 * Get USDC balance of an address on Base
 *
 * @param {string} address - Ethereum address
 * @returns {Promise<string>} Balance in USDC (formatted)
 */
async function getUSDCBalance(address) {
  try {
    const provider = getBaseProvider();
    const usdcContract = new ethers.Contract(USDC_BASE.address, ERC20_ABI, provider);
    const balance = await usdcContract.balanceOf(address);
    return ethers.formatUnits(balance, USDC_BASE.decimals);
  } catch (error) {
    throw new Error(`Failed to get USDC balance: ${error.message}`);
  }
}

/**
 * Format price from atomic units to USDC
 *
 * @param {string} atomicUnits - Price in atomic units (e.g., "1000" = $0.01)
 * @returns {string} Formatted USDC amount (e.g., "0.01")
 */
function formatPrice(atomicUnits) {
  // Our atomic units: 1000 = $0.01 USDC
  // Convert to USDC with 6 decimals: 1000 units * 10 = 10,000 USDC units = $0.01
  const usdcUnits = BigInt(atomicUnits) * BigInt(10);
  return ethers.formatUnits(usdcUnits, USDC_BASE.decimals);
}

/**
 * Get payment info for display
 */
function getPaymentInfo() {
  return {
    network: BASE_MAINNET,
    token: USDC_BASE,
    paymentWallet: PAYMENT_WALLET,
    instructions: {
      method: 'Send USDC on Base Mainnet',
      recipient: PAYMENT_WALLET,
      network: 'Base (Chain ID: 8453)',
      token: 'USDC (6 decimals)',
      confirmations: 'Wait for transaction to be mined',
      includeInPayment: 'Include transaction hash in X-PAYMENT header'
    }
  };
}

module.exports = {
  verifyPayment,
  getUSDCBalance,
  formatPrice,
  getPaymentInfo,
  BASE_MAINNET,
  USDC_BASE,
  PAYMENT_WALLET
};
