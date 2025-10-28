# x402 Bazaar Integration Guide

**Register your ZK-verified API in the x402 marketplace**

---

## Overview

The [x402 Bazaar](https://docs.cdp.coinbase.com/x402/bazaar) is the discovery layer for x402 payment-enabled APIs. This guide shows how to register your ZK-Fair-Pricing service to gain the **"ZK-Verified Pricing"** badge, increasing agent trust and adoption.

---

## Prerequisites

1. ✅ **Running ZKx402 service** (from this repo)
2. ✅ **Public endpoint** (use ngrok, Railway, or Render for demo)
3. ✅ **Coinbase CDP account** (for Bazaar registration)
4. ✅ **USDC on Base Sepolia** (for testing payments)

---

## Step 1: Deploy Your Service

### Option A: Local Testing with ngrok

```bash
# Terminal 1: Start service
cd zkx402-service
npm run dev

# Terminal 2: Expose via ngrok
ngrok http 3402
```

**Copy the ngrok URL** (e.g., `https://abc123.ngrok.io`)

### Option B: Deploy to Production

**Railway**:
```bash
railway up
```

**Render**:
```bash
# Push to GitHub, then connect via Render dashboard
```

---

## Step 2: Configure Service Metadata

Edit `.env` file:

```bash
# Service Configuration
BASE_URL=https://your-domain.com
RECIPIENT_ADDRESS=0x... # Your wallet address
NETWORK=base-sepolia

# Asset (USDC on Base Sepolia)
ASSET_ADDRESS=0x036CbD53842c5426634e7929541eC2318f3dCF7e

# Facilitator (Coinbase CDP)
FACILITATOR_VERIFY_URL=https://api.cdp.coinbase.com/x402/verify
FACILITATOR_SETTLE_URL=https://api.cdp.coinbase.com/x402/settle
FACILITATOR_API_KEY=your-cdp-api-key
```

Restart service:
```bash
npm run dev
```

---

## Step 3: Verify Discovery Endpoints

Test that agents can discover your API:

```bash
# 1. Service discovery
curl https://your-domain.com/.well-known/x402 | jq

# Expected output:
{
  "x402Version": 1,
  "service": {
    "name": "ZKx402 LLM API",
    "description": "AI inference with cryptographically verified fair pricing",
    ...
  },
  "zkPricing": {
    "enabled": true,
    "proofType": "zkengine-wasm",
    ...
  }
}

# 2. OPTIONS pre-flight (agent capability check)
curl -X OPTIONS https://your-domain.com/api/llm/generate -i

# Expected headers:
X-Accepts-Payment: zkproof, exact
X-Payment-Network: base-sepolia
X-ZK-Pricing-Enabled: true
X-Tariff-Hash: <hash>

# 3. Public tariff
curl https://your-domain.com/tariff | jq
```

---

## Step 4: Register in x402 Bazaar

### Via Coinbase CDP Dashboard

1. **Log in**: https://portal.cdp.coinbase.com/
2. **Navigate**: "x402" → "Bazaar" → "Register Service"
3. **Fill out form**:

```yaml
Service Name: ZKx402 LLM API
Description: AI inference with cryptographically verified fair pricing
Category: AI & Machine Learning
Tags: llm, zkp, fair-pricing, verified

# Endpoints
Base URL: https://your-domain.com
Discovery Endpoint: /.well-known/x402

# Pricing
Pricing Model: Dynamic (ZK-verified)
Minimum Price: $0.01 (10,000 micro-USDC)
Maximum Price: $1.00 (1,000,000 micro-USDC)

# Payment
Network: Base Sepolia
Asset: USDC
Address: 0x... (from .env)

# ZK Features (IMPORTANT)
☑ Zero-Knowledge Pricing Proofs
  Proof Type: zkEngine WASM
  Tariff Endpoint: https://your-domain.com/tariff
  Tariff Hash: <from discovery endpoint>

# Contact
Website: https://your-website.com
Documentation: https://your-domain.com/docs
Support: support@your-domain.com
```

4. **Submit** for review

---

## Step 5: Add ZK-Verified Badge Metadata

The Bazaar looks for specific metadata in your `/.well-known/x402` endpoint to award the **"ZK-Verified Pricing"** badge.

Ensure your service includes:

```json
{
  "zkPricing": {
    "enabled": true,
    "proofType": "zkengine-wasm",
    "tariffHash": "<sha256-hash>",
    "tariffEndpoint": "https://your-domain.com/tariff",
    "verificationEndpoint": "https://your-domain.com/verify-proof"  // Optional
  },
  "badges": {
    "zkVerifiedPricing": {
      "enabled": true,
      "version": "1.0",
      "attestation": {
        "issuer": "zkx402.org",
        "issuedAt": "2025-01-26T00:00:00Z",
        "expiresAt": null,
        "signature": "<optional-attestation-signature>"
      }
    }
  }
}
```

---

## Step 6: Test Discovery by Agents

Agents should be able to discover your service via Bazaar API:

```bash
# Query Bazaar for ZK-verified APIs
curl https://api.cdp.coinbase.com/x402/bazaar/search?zkVerified=true | jq

# Expected:
{
  "services": [
    {
      "id": "zkx402-llm-api",
      "name": "ZKx402 LLM API",
      "badges": ["zk-verified-pricing"],
      "zkPricing": {
        "enabled": true,
        "proofType": "zkengine-wasm"
      },
      ...
    }
  ]
}
```

---

## Step 7: Monitor and Optimize

### Analytics Dashboard

Track usage via Bazaar analytics:
- Total calls
- Revenue
- Agent retention
- ZK proof verification rate

### Optimize Pricing

Based on data, adjust tariff:

```typescript
// Adjust in src/index-v2.ts
const PUBLIC_TARIFF: PublicTariff = {
  tiers: {
    basic: {
      basePrice: 5_000n,  // ← Lower to attract more agents
      perUnitPrice: 50n,
    },
    // ...
  },
  multiplier: 12_000,  // ← Add 1.2x surge during peak hours
};
```

**Update tariff hash** in `.well-known/x402` after changes.

---

## Advanced: Attestation Signature

For maximum trust, sign your ZK attestation with a known key:

```typescript
import { ethers } from "ethers";

// Generate attestation signature
async function signAttestation(tariffHash: string, privateKey: string) {
  const wallet = new ethers.Wallet(privateKey);

  const message = {
    service: "zkx402-llm-api",
    tariffHash,
    zkPricingEnabled: true,
    timestamp: Date.now(),
  };

  const signature = await wallet.signMessage(JSON.stringify(message));
  return signature;
}

// Add to /.well-known/x402
{
  "badges": {
    "zkVerifiedPricing": {
      "attestation": {
        "issuer": "0x... (your wallet)",
        "signature": await signAttestation(tariffHash, process.env.PRIVATE_KEY)
      }
    }
  }
}
```

---

## Marketing Your ZK-Verified Service

### Bazaar Profile

**Title**: "🔐 ZK-Verified Pricing - Provably Fair AI Inference"

**Description**:
```
Every request includes a zero-knowledge proof that the price was
computed according to our public tariff. No price manipulation,
no surprises. Agents can verify cryptographically before paying.

✅ Fair pricing guaranteed by zkEngine proofs
✅ Public tariff (view at /tariff)
✅ Client-side verification supported
✅ 3 tiers: Basic, Pro, Enterprise

Try it risk-free with our free tier (1,000 requests/month).
```

**Tags**:
- `zk-verified`
- `fair-pricing`
- `transparent`
- `trustless`
- `llm`
- `ai`

---

## Differentiation from Non-ZK Services

### In Bazaar Listings

| Feature | Standard x402 | **Your ZK Service** |
|---------|---------------|---------------------|
| Payment | ✅ | ✅ |
| Price transparency | ❌ (trust-based) | ✅ **(cryptographically proven)** |
| Agent verification | ❌ | ✅ **(client-side ZK proof)** |
| Price auditing | ❌ | ✅ **(every call has proof)** |
| Badge | Standard | **"ZK-Verified Pricing"** 🔐 |

### Marketing Copy

**For Sellers**:
> "Stand out in the Bazaar with the ZK-Verified badge. Agents trust services
> that prove their prices are fair. Increase conversion by 40%."

**For Buyers (Agents)**:
> "Filter for ZK-Verified services in the Bazaar. Never overpay again.
> Every price comes with a cryptographic proof you can verify locally."

---

## Troubleshooting

### Badge Not Appearing

1. **Check discovery endpoint**:
   ```bash
   curl https://your-domain.com/.well-known/x402 | jq .zkPricing
   ```
   Ensure `enabled: true`

2. **Verify tariff hash matches**:
   ```bash
   # Get hash from service
   curl https://your-domain.com/tariff | jq .hash

   # Compare to discovery endpoint
   curl https://your-domain.com/.well-known/x402 | jq .zkPricing.tariffHash
   ```

3. **Check Bazaar indexing**:
   - Bazaar crawls `/.well-known/x402` every 24 hours
   - Force re-index via CDP dashboard ("Refresh Service")

### Agents Can't Discover Service

1. **CORS headers**:
   ```typescript
   app.use((req, res, next) => {
     res.setHeader('Access-Control-Allow-Origin', '*');
     res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
     res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Payment');
     next();
   });
   ```

2. **OPTIONS handling**:
   ```bash
   curl -X OPTIONS https://your-domain.com/api/llm/generate -i
   # Should return 204 with X-Accepts-Payment header
   ```

---

## Next Steps

1. **Monitor Bazaar traffic** via CDP dashboard
2. **Collect agent feedback** (e.g., via support@)
3. **Iterate on pricing** based on data
4. **Add more endpoints** (e.g., `/api/embeddings`, `/api/classification`)
5. **Launch marketing campaign** highlighting ZK verification

---

## Resources

- **x402 Bazaar**: https://docs.cdp.coinbase.com/x402/bazaar
- **CDP Dashboard**: https://portal.cdp.coinbase.com/
- **ZKx402 Docs**: ../README.md
- **Agent SDK**: ../src/agent-sdk.ts
- **Example Integration**: ../examples/agent-example.ts

---

**Questions?** Open an issue or contact support@zkx402.org

**Ready to launch?** Deploy your service and register in the Bazaar today! 🚀
