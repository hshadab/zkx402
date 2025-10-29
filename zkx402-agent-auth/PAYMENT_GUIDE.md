# zkX402 Payment Guide - Base USDC Integration

## Overview

zkX402 now accepts **real stablecoin payments** on **Base L2** for agent authorization proofs. All x402 protocol requests require payment in USDC to access zkML proof generation services.

## Production API Endpoints

**Production URL**: https://zk-x402.com

All examples below use the production domain. For local development, replace with `http://localhost:3001`.

## Payment Details

### Network Information
- **Blockchain**: Base Mainnet (Ethereum L2)
- **Chain ID**: 8453
- **RPC URL**: `https://mainnet.base.org`
- **Explorer**: [https://basescan.org](https://basescan.org)

### Payment Token
- **Token**: USDC (USD Coin)
- **Contract Address**: `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`
- **Decimals**: 6
- **Standard**: ERC-20

### Payment Wallet
- **Recipient Address**: `0x1f409E94684804e5158561090Ced8941B47B0CC6`
- **Explorer**: [View on BaseScan](https://basescan.org/address/0x1f409E94684804e5158561090Ced8941B47B0CC6)

## Pricing

All prices are listed in atomic units (1000 units = $0.01 USDC):

### Production Models
| Model | Price (USDC) | Use Case |
|-------|--------------|----------|
| **simple_threshold** | $0.01 | Basic balance check |
| **percentage_limit** | $0.015 | Percentage-based spending limits |
| **vendor_trust** | $0.01 | Marketplace vendor verification |
| **velocity_1h** | $0.02 | Hourly rate limiting |
| **velocity_24h** | $0.02 | Daily spending caps |
| **daily_limit** | $0.02 | Budget enforcement |
| **age_gate** | $0.01 | Age-restricted purchases |
| **multi_factor** | $0.05 | High-security transactions |
| **composite_scoring** | $0.04 | Advanced risk assessment |
| **risk_neural** | $0.06 | Sophisticated fraud detection |

### Test Models
| Model | Price (USDC) | Use Case |
|-------|--------------|----------|
| **test_less** | $0.005 | Operation verification |
| **test_identity** | $0.005 | Pass-through testing |
| **test_clip** | $0.005 | Activation function testing |
| **test_slice** | $0.005 | Tensor operation testing |

## Payment Flow

### For AI Agents (x402 Protocol)

1. **Discover Payment Requirements**
   ```bash
   # Production
   curl https://zk-x402.com/.well-known/x402

   # Local development
   curl http://localhost:3001/.well-known/x402
   ```
   Response includes payment details:
   ```json
   {
     "payment": {
       "enabled": true,
       "blockchain": "Base Mainnet",
       "chainId": 8453,
       "token": "USDC",
       "tokenAddress": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
       "paymentWallet": "0x1f409E94684804e5158561090Ced8941B47B0CC6"
     }
   }
   ```

2. **Request Authorization (402 Response)**
   ```bash
   # Production
   curl -X POST https://zk-x402.com/x402/authorize/simple_threshold

   # Local development
   curl -X POST http://localhost:3001/x402/authorize/simple_threshold
   ```
   Returns HTTP 402 with payment requirements:
   ```json
   {
     "x402Version": 1,
     "accepts": [{
       "scheme": "zkml-jolt",
       "network": "base-mainnet",
       "asset": "USDC",
       "payTo": "0x1f409E94684804e5158561090Ced8941B47B0CC6",
       "maxAmountRequired": "1000",
       "payment": {
         "blockchain": "Base",
         "chainId": 8453,
         "token": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
         "tokenSymbol": "USDC",
         "amountUSDC": "0.01",
         "instructions": "Send USDC on Base network, then include transaction hash in X-PAYMENT header with zkML proof"
       }
     }]
   }
   ```

3. **Send USDC Payment on Base**
   - Transfer required USDC amount to payment wallet
   - Wait for transaction confirmation
   - Save transaction hash

4. **Generate zkML Proof**
   - Generate JOLT Atlas proof for your authorization inputs
   - Combine proof with payment transaction hash

5. **Submit Payment + Proof**
   ```bash
   # Production
   curl -X POST https://zk-x402.com/x402/authorize/simple_threshold \
     -H "Content-Type: application/json" \
     -H "X-PAYMENT: <base64-encoded-payload>"

   # Local development
   curl -X POST http://localhost:3001/x402/authorize/simple_threshold \
     -H "Content-Type: application/json" \
     -H "X-PAYMENT: <base64-encoded-payload>"
   ```

   Payload structure (before base64 encoding):
   {
     "x402Version": 1,
     "scheme": "zkml-jolt",
     "network": "base-mainnet",
     "payload": {
       "modelId": "simple_threshold",
       "paymentTxHash": "0x123...",
       "zkmlProof": {
         "approved": true,
         "output": [...],
         "verification": {...}
       }
     }
   }
   ```

6. **Receive Authorization**
   - Server verifies payment on-chain
   - Server verifies zkML proof
   - Returns authorization result with X-PAYMENT-RESPONSE header

## Payment Verification

The server performs the following checks:

1. ✅ **Transaction exists** on Base mainnet
2. ✅ **Transaction succeeded** (not reverted)
3. ✅ **Payment amount** matches or exceeds required price
4. ✅ **Recipient address** matches payment wallet
5. ✅ **Token contract** is USDC on Base
6. ✅ **Transaction age** < 10 minutes (freshness check)
7. ✅ **zkML proof** structure and content valid

## Example: Web3 Integration

```javascript
const { ethers } = require('ethers');

// Connect to Base
const provider = new ethers.JsonRpcProvider('https://mainnet.base.org');
const wallet = new ethers.Wallet(PRIVATE_KEY, provider);

// USDC contract on Base
const USDC_ADDRESS = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';
const PAYMENT_WALLET = '0x1f409E94684804e5158561090Ced8941B47B0CC6';

const usdcAbi = [
  'function transfer(address to, uint256 amount) returns (bool)'
];

const usdcContract = new ethers.Contract(USDC_ADDRESS, usdcAbi, wallet);

// Send payment for simple_threshold model ($0.01 = 10,000 USDC units with 6 decimals)
const amount = 10000; // 0.01 USDC
const tx = await usdcContract.transfer(PAYMENT_WALLET, amount);
const receipt = await tx.wait();

console.log('Payment transaction:', receipt.hash);

// Now use receipt.hash in X-PAYMENT header
```

## Free Tier vs Paid Tier

### Free Tier (UI Only)
- **Limit**: 5 proof generations per day
- **Access**: Web UI at https://zk-x402.com
- **Purpose**: Testing and demonstrations
- **No payment required**: Rate-limited by IP

### Paid Tier (x402 Protocol)
- **Limit**: Unlimited
- **Access**: API via x402 protocol
- **Purpose**: Production AI agents
- **Payment required**: USDC on Base per proof

## API Endpoints

### Get Payment Information
```bash
GET /x402/payment-info
```

Returns complete payment details including network, token, and wallet information.

### List Models with Pricing
```bash
GET /x402/models
```

Returns all models with pricing in both atomic units and USDC.

### Check Service Status
```bash
GET /.well-known/x402
```

Returns service discovery information including payment configuration.

## Security Notes

1. **Transaction Hash Uniqueness**: Each payment transaction can only be used once
2. **Time-based Expiry**: Transactions older than 10 minutes are rejected
3. **Amount Verification**: Exact or higher payment amounts are accepted
4. **On-chain Verification**: All payments are verified directly on Base blockchain
5. **No Refunds**: Failed proofs are not refunded (payment covers computational cost)

## Support

For payment issues or questions:
- **GitHub**: [https://github.com/hshadab/zkx402/issues](https://github.com/hshadab/zkx402/issues)
- **Documentation**: [https://github.com/hshadab/zkx402](https://github.com/hshadab/zkx402)

## Base Network Resources

- **Add Base to MetaMask**: [https://bridge.base.org/](https://bridge.base.org/)
- **Bridge to Base**: [https://bridge.base.org/](https://bridge.base.org/)
- **Get USDC**: Bridge from Ethereum mainnet or buy on Base
- **Faucet (Testnet)**: Not applicable - mainnet only

---

**Last Updated**: 2025-10-29
**Payment Integration Version**: 1.0.0
