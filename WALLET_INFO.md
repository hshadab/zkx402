# Payment Wallet Information

## ℹ️ Overview

Your zkX402 app receives USDC payments on Base Mainnet. This document tells you where to monitor payments and whether you need to fund the wallet.

---

## 💰 Your Payment Wallet Address

**Send USDC payments to:**

```
0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
```

**Network:** Base Mainnet (Chain ID: 8453)

**View on BaseScan:** https://basescan.org/address/0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91

---

## ❓ Do You Need to Send Funds?

### **NO - You Don't Need to Fund This Wallet**

Your app **receives** USDC payments from users. The workflow is:

1. User wants to use zkX402 service
2. User sends USDC to your wallet address (above)
3. Your app verifies the payment on-chain
4. Your app generates the zkML proof
5. USDC stays in your wallet

**Your wallet is RECEIVE-ONLY** - you don't need to send any funds to it initially.

---

## 🔧 When You DO Need to Add Funds

You only need to add ETH (for gas fees) if you want to **withdraw** the accumulated USDC:

**Scenario:** You've received 100 USDC in payments and want to move it to your personal wallet.

**What you need:**
- ~0.001 ETH on Base (~$3-5 USD)
- This covers gas fees for the withdrawal transaction

**How to add ETH:**
1. Bridge from Ethereum: https://bridge.base.org
2. Or use Coinbase (supports Base deposits)

---

## 📊 Monitoring Your Payments

### View USDC Balance

**BaseScan (Easiest):**
1. Go to: https://basescan.org/address/0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
2. Click "Token" tab
3. See USDC balance

**Using ethers.js:**
```javascript
const ethers = require('ethers');
const provider = new ethers.JsonRpcProvider('https://mainnet.base.org');
const USDC_ADDRESS = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';

const usdcContract = new ethers.Contract(
  USDC_ADDRESS,
  ['function balanceOf(address) view returns (uint256)'],
  provider
);

usdcContract.balanceOf('0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91').then(balance => {
  const usdcAmount = ethers.formatUnits(balance, 6);
  console.log(`Balance: ${usdcAmount} USDC`);
});
```

### View Recent Transactions

**BaseScan:**
- Go to wallet address
- See all incoming USDC transfers
- Click transaction hash for details

---

## 🔐 Security Notes

**Private Key Storage:**
- Private key is in `zkx402-agent-auth/ui/.env` (git-ignored)
- Also add as `PAYMENT_WALLET` environment variable on Render
- Never commit .env to git (already in .gitignore)

**Wallet Security:**
- This wallet was generated in an AI conversation (lower security)
- ✅ Safe for receiving test payments and small amounts
- ⚠️ For production with significant funds, consider creating new wallet on your own machine
- 💡 Withdraw funds regularly to cold storage/hardware wallet

---

## 📝 Configuration

### Render.com Environment Variable

Set this on your Render dashboard:

```
PAYMENT_WALLET=0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
```

**URL:** https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0

**Steps:**
1. Go to service → Environment tab
2. Add variable: `PAYMENT_WALLET`
3. Save (auto-redeploys)

### Local Development

The `.env` file already contains your wallet address and private key.

---

## 🧪 Testing Payments

### Send Test Payment

1. Use MetaMask or any wallet
2. Switch to Base Mainnet
3. Send 0.01 USDC to: `0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91`
4. Wait for confirmation (~2 seconds)
5. Check BaseScan to see it arrived

### Verify in Your App

Your app will verify payments automatically when users make requests with the x402 protocol.

---

## 💸 Withdrawing Funds (When Needed)

### Option 1: MetaMask (Easiest)

1. Import wallet to MetaMask using private key from `.env`
2. Add ~0.001 ETH to wallet for gas
3. Send USDC to your personal wallet

### Option 2: Script

See `WALLET_SETUP.md` for withdrawal scripts.

---

## 📋 Summary

**What you need to do NOW:**
- ✅ Nothing! Wallet is configured and ready to receive

**What you need to do LATER (when you have payments):**
- Add ~0.001 ETH for gas fees (only if withdrawing)
- Withdraw USDC to personal wallet regularly

**What you should monitor:**
- Check BaseScan occasionally to see incoming payments
- Withdraw accumulated USDC periodically

---

## 🆘 Support

**View wallet:** https://basescan.org/address/0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91

**Base Network Status:** https://status.base.org

**Private key:** Check `zkx402-agent-auth/ui/.env` (never share this!)

---

**TL;DR:** Your wallet is `0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91` on Base Mainnet. You don't need to fund it - it receives USDC from users. Only add ETH later if you want to withdraw.
