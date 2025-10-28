# ZKx402 Quick Start Guide

**Get running in 5 minutes** ⚡

---

## Prerequisites

- Rust (1.70+) - [Install](https://rustup.rs/)
- Node.js (20+) - [Install](https://nodejs.org/)
- `curl` and `jq` (for testing)

---

## Step 1: Clone the Project ✅

You already have this! You're in `/home/hshadab/zkx402`.

---

## Step 2: Generate Your First ZK Proof 🔐

```bash
cd zkEngine_dev
cargo run --release --example zkx402_pricing
```

**What you'll see**:
```
=== ZKx402 Fair-Pricing Proof Generator ===

Public Tariff:
  Basic:      $0.0100 base + $0.000100/token
  Pro:        $0.0500 base + $0.000350/token
  Enterprise: $0.1000 base + $0.000800/token

Request Metadata:
  Tokens: 1200
  Tier: 1 (Pro)

Expected price: $0.564000

[1/4] Generating public parameters (step_size=50)...
      ✓ Setup complete

[2/4] Building WASM execution context...
      ✓ Context built

[3/4] Generating zero-knowledge proof...
      ✓ Proof generated

[4/4] Verifying proof...
      ✓ Proof verified successfully!

✅ Zero-knowledge proof confirms:
   The price $0.564000 was computed correctly
   according to the public tariff.
```

**⏱️ Time**: ~10-15 seconds (first run compiles Rust; subsequent runs ~5s)

---

## Step 3: Start the x402 Service 🚀

```bash
cd ../zkx402-service
npm install
npm run dev
```

**What you'll see**:
```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔐 ZKx402 Fair-Pricing Service                             ║
║                                                               ║
║   Zero-Knowledge Proofs for x402 Protocol                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Server running on: http://localhost:3402

Public Tariff:
  Basic:      $0.0100 base + $0.000100/token
  Pro:        $0.0500 base + $0.000350/token
  Enterprise: $0.1000 base + $0.000800/token
```

**⏱️ Time**: ~30 seconds (npm install)

---

## Step 4: Test the API 🧪

Open a new terminal and run:

```bash
# View the public tariff
curl http://localhost:3402/tariff | jq

# Make a request WITHOUT payment (get 402 + ZK proof)
curl -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is zero-knowledge?","tier":1}' | jq
```

**Expected response**:
```json
{
  "error": "Payment Required",
  "details": {
    "price": "56400",
    "chain": "base-sepolia",
    "asset": "usdc",
    "zkProof": {
      "type": "fair-pricing",
      "verified": true,
      "message": "This price has been cryptographically proven to match the public tariff"
    }
  }
}
```

**Look at the headers**:
```bash
curl -i -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test","tier":1}'
```

You'll see:
```
HTTP/1.1 402 Payment Required
X-Accept-Payment: base-sepolia:usdc:51400
X-Pricing-Proof: {"proof":"...","type":"zkengine-wasm",...}
```

---

## Step 5: Simulate a Paid Request 💳

```bash
# Retry with X-PAYMENT header (simulates payment)
curl -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -H "X-PAYMENT: mock-payment-token" \
  -d '{"prompt":"What is zero-knowledge?","tier":1}' | jq
```

**Response**:
```json
{
  "result": "Mock LLM response to: \"What is zero-knowledge?\"",
  "usage": {
    "promptTokens": 5,
    "completionTokens": 50,
    "totalTokens": 55
  },
  "zkx402": {
    "message": "Payment verified. This response was paid for with x402 + ZK proof."
  }
}
```

---

## What Just Happened? 🤔

1. **Agent** sent a request without payment
2. **Server** computed the fair price based on public tariff
3. **zkEngine** generated a ZK proof that price = f(tokens, tier, tariff)
4. **Server** returned 402 with proof in headers
5. **Agent** (you) verified the proof matches the tariff
6. **Agent** paid (simulated with `X-PAYMENT` header)
7. **Server** delivered the response

**The ZK proof guarantees** the server didn't price-gouge!

---

## Try Different Tiers 🎚️

```bash
# Basic tier (cheaper)
curl -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Short prompt","tier":0}' | jq .details.price

# Pro tier
curl -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Medium prompt","tier":1}' | jq .details.price

# Enterprise tier (most expensive)
curl -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Long prompt","tier":2}' | jq .details.price
```

Notice the prices change based on:
- **Prompt length** (estimated tokens)
- **Tier** (basic/pro/enterprise)

---

## Run the Full Demo 🎬

```bash
cd zkx402-service
./test-demo.sh
```

This runs a complete end-to-end test with colored output showing:
1. Tariff fetch
2. 402 challenge + proof
3. Payment + response

---

## Next Steps 📚

### Learn More

- **[README.md](README.md)** - Full documentation + business model
- **[TECHNICAL_SPEC.md](TECHNICAL_SPEC.md)** - Crypto protocol deep dive
- **[SUMMARY.md](SUMMARY.md)** - Project overview + competitive analysis

### Customize the Tariff

Edit `zkx402-service/src/index.ts`:
```typescript
const PUBLIC_TARIFF: PublicTariff = {
  tiers: {
    basic: {
      basePrice: 10_000n,  // Change this!
      perUnitPrice: 100n,
    },
    // ...
  },
  multiplier: 10_000,  // Or add surge pricing here (e.g., 15000 = 1.5x)
};
```

### Build Your Own ZK Circuit

1. Edit `zkEngine_dev/wasm/zkx402/pricing.wat`
2. Add your custom pricing logic (volume discounts, credits, etc.)
3. Test with `cargo run --example zkx402_pricing`
4. Integrate into the service

### Deploy to Production

See `TECHNICAL_SPEC.md` Section 7.1 for deployment guide.

---

## Troubleshooting 🔧

### "Rust compiler not found"
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### "npm: command not found"
Install Node.js from https://nodejs.org/

### Port 3402 already in use
```bash
# Kill existing process
lsof -ti:3402 | xargs kill -9

# Or use a different port
PORT=3403 npm run dev
```

### Proof generation takes too long
This is normal on first run (compiles Rust). Subsequent runs are ~5s.

If still slow, try:
```bash
cargo build --release  # Pre-compile
cargo run --release --example zkx402_pricing  # Now it's fast
```

---

## You're All Set! 🎉

You now have a **working ZK-Fair-Pricing service** with:
- ✅ Real zero-knowledge proofs (zkEngine)
- ✅ x402 payment integration
- ✅ Express REST API
- ✅ Public tariff system

**Time to ship it!** 🚀

---

**Questions?** Open an issue or check the [README](README.md).
