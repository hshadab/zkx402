# ZKx402: Complete Trust Stack for Agent Commerce

**Zero-Knowledge Proofs for Fair Pricing + Agent Authorization**

Two complementary ZK services that solve both sides of the trust problem in AI agent payments:

---

## 📖 What This Does (Plain English)

AI agents are buying API services, but there are **two trust problems**:

### Problem 1: Can you trust the seller's price? 💰
**Imagine a restaurant that can secretly change the bill after you eat.** That's current AI agent payments.

**ZK-Fair-Pricing solves this** with a cryptographic proof that:
- ✅ The price was computed correctly per public tariff
- ✅ No price manipulation occurred
- ✅ You can verify instantly without trusting the seller

### Problem 2: Can the seller trust the agent is authorized? 🤖
**Imagine a company credit card with no spending limits.** Agents could overspend or violate policies.

**ZK-Agent-Authorization solves this** with a cryptographic proof that:
- ✅ The agent is authorized to spend this amount
- ✅ No policy violations (budget, velocity, whitelist, etc.)
- ✅ Privacy preserved (balance and limits stay hidden)

### Together: Complete Trust 🔐
Both proofs stack on the same transaction:
- **Buyer trusts**: "The price is fair" (Fair-Pricing proof)
- **Seller trusts**: "The agent is authorized" (Agent-Auth proof)
- **Both get**: Cryptographic guarantees, no disputes, full audit trail

### Real-World Example

**Without ZKx402**:
```
You: "Hey AI, write me a blog post"
AI API: "That'll be $0.50"
You: "Okay... but how do I know that's the right price?"
AI API: "Just trust me 🤷"
```

**With ZKx402**:
```
You: "Hey AI, write me a blog post"
AI API: "That'll be $0.50. Here's a cryptographic proof that shows:
        - Your request was 1,200 tokens
        - Pro tier costs $0.05 base + $0.00035 per token
        - Math: $0.05 + (1,200 × $0.00035) = $0.47
        - No surge pricing today, so final = $0.47

        You can verify this yourself in milliseconds!"
You: "✅ Verified! Here's payment."
```

### Who This Helps

- **AI Agents** 🤖 - Can't be overcharged (Fair-Pricing) AND can't overspend (Agent-Auth)
- **API Sellers** 💼 - Prove honest pricing AND verify agent authorization
- **Enterprises** 🏢 - Audit trails + compliance (SOC2, GDPR, SEC)
- **Developers** 👨‍💻 - Build fully trustless agent commerce

---

## 🎯 What This Is (Technical)

**ZKx402** adds **zero-knowledge proofs** to the [x402 payment protocol](https://docs.cdp.coinbase.com/x402) to solve its biggest trust problem: **price verification**.

- **x402 today**: Agents pay per API call, but must *trust* the seller set the right price
- **ZKx402**: Every payment includes a **cryptographic proof** that the price matches the public tariff

### The Problem

Current x402 flow:
```http
GET /api/data
HTTP/1.1 402 Payment Required
X-ACCEPT-PAYMENT: base-sepolia:usdc:0.50
```

**Critical gap**: The server *claims* the price is $0.50, but **nothing proves** it was calculated correctly. Agents can't verify they weren't price-gouged without manual reconciliation.

### The Solution

ZKx402 flow:
```http
GET /api/data
HTTP/1.1 402 Payment Required
X-ACCEPT-PAYMENT: base-sepolia:usdc:564000
X-PRICING-PROOF: {"proof": "...", "type": "zkengine-wasm", ...}
```

**Agents can now verify**: The proof cryptographically guarantees that `564000 micro-USDC = compute_price(1200 tokens, tier=Pro, PUBLIC_TARIFF)`.

---

## 🏗️ What's Inside

### Use Case #1: ZK-Fair-Pricing (✅ Built)

#### 1.1 **Real ZK Proofs** (`zkEngine_dev/`)

- **WASM circuit** for price computation (`wasm/zkx402/pricing.wat`)
- **Rust prover** using [zkEngine](https://github.com/ICME-Lab/zkEngine_dev) (Nebula NIVC zkWASM)
- **Example**: Run `cargo run --release --example zkx402_pricing` to see a real proof generated in ~5-10s

```rust
// Proves: final_price = compute_price(tokens, tier, tariff, multiplier)
let (snark, instance) = WasmSNARK::<E, S1, S2>::prove(&pp, &wasm_ctx, step_size)?;
snark.verify(&pp, &instance)?; // ✅ Verified!
```

#### 1.2 **x402 Middleware** (`zkx402-service/`)

- **Express middleware** that auto-generates ZK proofs for 402 challenges
- **Public tariff** system (basic/pro/enterprise tiers)
- **TypeScript service** with REST API

```typescript
app.post('/api/llm/generate',
  zkx402Middleware({
    tariff: PUBLIC_TARIFF,
    computeMetadata: (req) => ({
      tokens: estimateTokens(req.body.prompt),
      tier: req.user?.tier || 0,
    }),
    facilitator: { chain: 'base-sepolia', asset: 'usdc' }
  }),
  async (req, res) => {
    // Only runs if payment verified
    res.json({ result: '...' });
  }
);
```

### Use Case #2: ZK-Agent-Authorization (✅ Built)

#### 2.1 **Hybrid Proof System** (`zkx402-agent-auth/`)

Two backends for different policy complexities:

**JOLT Atlas** - Simple policies (velocity, trust scoring) via ONNX neural networks
- Proving time: ~0.7s
- Proof size: 524 bytes
- Use case: Numeric comparisons, ML-based policies
- Technology: [JOLT Atlas zkML](https://github.com/ICME-Lab/jolt-atlas) with ONNX models

**zkEngine WASM** - Complex policies (whitelist, business hours, multi-condition)
- Proving time: ~5-10s
- Proof size: ~1-2KB
- Use case: IF/ELSE logic, string ops, time-based rules
- Technology: [zkEngine](https://github.com/ICME-Lab/zkEngine_dev) Nebula NIVC

#### 2.2 **Policy Examples**

Simple velocity policy (JOLT Atlas with ONNX):
```python
# Train policy as neural network
class VelocityPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 16),  # 5 inputs: amount, balance, vel_1h, vel_24h, trust
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),   # 2 outputs: approved_score, risk_score
            nn.Sigmoid()
        )
```

```rust
// Generate ZK proof with JOLT Atlas
let model = model(&"velocity_policy.onnx".into());
let snark = JoltSNARK::prove(pp, execution_trace, &program_output);
snark.verify((&pp).into(), program_output)?;  // ✅ Verified!
```

Complex whitelist + business hours policy (zkEngine WASM):
```wasm
(func (export "check_authorization")
    (param $transaction_amount i64)
    (param $vendor_id i64)
    (param $timestamp i64)
    (param $whitelist_bitmap i64)
    (result i64)  ;; authorized: 0 or 1

    ;; Check whitelist, business hours, budget...
    ;; Returns 1 if authorized, 0 otherwise
)
```

#### 2.3 **Hybrid Router** (`hybrid-router/`)

TypeScript service that auto-selects the right backend:

```typescript
// Classify policy complexity
const classification = PolicyClassifier.classify(request);
// → { backend: "jolt", reason: "Simple numeric policy" }

// Route to appropriate prover
if (classification.backend === "jolt") {
    proof = await joltClient.generateProof(request);
} else {
    proof = await zkEngineClient.generateProof(request);
}
```

---

## 🚀 Quick Start

### Prerequisites

- **Rust** (1.70+) for zkEngine + JOLT
- **Node.js** (20+) for TypeScript services
- **Python** (3.9+) for ONNX model training
- **Cargo** and **npm**

### Option A: Fair-Pricing Only (Quickest)

#### 1. Generate a Real Fair-Pricing ZK Proof

```bash
cd zkEngine_dev
cargo run --release --example zkx402_pricing
```

**Output**:
```
[3/4] Generating zero-knowledge proof...
      (This proves: final_price = compute_price(metadata, tariff))
      ✓ Proof generated

✅ Zero-knowledge proof confirms:
   The price $0.564000 was computed correctly
   according to the public tariff.
```

#### 2. Run the x402 Service

```bash
cd zkx402-service
npm install
npm run dev
```

**Server starts on `http://localhost:3402`**

#### 3. Test the API

```bash
# Get the public tariff
curl http://localhost:3402/tariff

# Make a request (no payment) -> Get 402 + ZK proof
curl -X POST http://localhost:3402/api/llm/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, world!","tier":1}'
```

**Response**:
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

**Headers**:
```
X-Accept-Payment: base-sepolia:usdc:56400
X-Pricing-Proof: {"proof":"...","type":"zkengine-wasm",...}
```

### Option B: Complete Trust Stack (Both Proofs)

See [DEMO_COMBINED.md](DEMO_COMBINED.md) for full walkthrough.

#### 1. Start Auth Router

```bash
cd zkx402-agent-auth/hybrid-router
npm install
npm run dev
# Server starts on http://localhost:3403
```

#### 2. Test Agent-Auth Proof Generation

```bash
# Train ONNX policy model first
cd zkx402-agent-auth/policy-examples/onnx
pip install -r requirements.txt
python train_velocity.py

# Simple policy (JOLT Atlas - real ONNX proof)
cd ../../jolt-prover
cargo run --release --example velocity_auth

# Complex policy (zkEngine - real WASM proof)
cd ../zkengine-prover
cargo run --release --example complex_auth
```

#### 3. Run Combined Demo

```bash
cd zkx402-service
tsx examples/demo-combined.ts
```

This demo shows Fair-Pricing + Agent-Auth proofs stacked on the same transaction!

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent (Caller)                       │
│  1. Send request                                            │
│  2. Receive 402 + ZK proof                                  │
│  3. Verify proof locally                                    │
│  4. Pay if valid                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    ZKx402 Service (Seller)                  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Express Middleware                                 │   │
│  │  1. Extract request metadata (tokens, tier)        │   │
│  │  2. Call ZK Prover                                  │   │
│  │  3. Generate 402 response with proof               │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ZK Prover (Rust zkEngine)                         │   │
│  │  1. Build WASM context                             │   │
│  │  2. Prove: price = f(tokens, tier, tariff)         │   │
│  │  3. Return serialized SNARK                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Public Tariff (On-Chain?)                  │
│  - Basic:      $0.01 + $0.0001/token                        │
│  - Pro:        $0.05 + $0.00035/token                       │
│  - Enterprise: $0.10 + $0.0008/token                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Technical Details

### Pricing Circuit (WASM)

**File**: `zkEngine_dev/wasm/zkx402/pricing.wat`

**Computation**:
```wasm
compute_price(tokens, tier, tariff, multiplier) -> price

subtotal = base_price[tier] + (tokens * per_unit_price[tier])
final_price = (subtotal * multiplier) / 10000
```

**Opcodes**: ~50 (for `compute_price` function)
**Step size**: 50
**Proving time**: ~5-10s (single-threaded, release build)
**Verification time**: <100ms

### zkEngine Backend

- **Proving system**: [Nebula](https://eprint.iacr.org/2024/1605) (NIVC zkWASM)
- **Curve cycle**: Bn256 + Grumpkin (IPA-based polynomial commitment)
- **SNARK type**: Recursive SNARK (can be compressed to Groth16 for on-chain verification)
- **Memory**: Handles ~147KB linear memory efficiently with `set_memory_step_size(50_000)`

### Public Inputs

The proof reveals:
- ✅ `tokens` (request size)
- ✅ `tier` (pricing tier)
- ✅ `tariff` (public rate card)
- ✅ `final_price` (computed result)

The proof hides:
- ❌ User identity (no wallet address in proof)
- ❌ Request content (only metadata hash if desired)

---

## 💰 Business Model

### Pricing Tiers

| Plan | Price per Proof | Volume | Use Case |
|------|----------------|--------|----------|
| **Free** | $0 | 1,000/month | Dev & testing |
| **Startup** | $0.003 | 50K/month | Small APIs |
| **Scale** | $0.001 | 500K/month | Production APIs |
| **Enterprise** | Custom | Unlimited | White-label, custom models |

### Revenue Model

1. **SaaS**: Charge per proof generated
2. **Marketplace fee**: 1% of x402 payment volume for "ZK-verified" badge
3. **White-label**: License to x402 facilitators (Coinbase, etc.) at $5K-10K/year

### Unit Economics

- **Cost**: 1 CPU core = ~720 proofs/hour (5s per proof) = 17K/day
- **AWS c6i.2xlarge** (8 cores): $245/month → 136K proofs/day
- **Revenue**: 136K × $0.001 = $136/day = $4,080/month
- **Profit**: $4,080 - $245 = **$3,835/month per server**

**At scale (10M proofs/month)**: ~$10K revenue, ~$2K infra cost = **$8K profit**

---

## 🎯 Why This Adds Value to x402

### Current x402 (without ZK)

- ✅ Payment rail (agents can pay per call)
- ❌ **No price verification** (must trust seller)
- ❌ No privacy (facilitator sees payer)
- ❌ No abuse control (free tiers = captchas)

### ZKx402 (with ZK)

- ✅ Payment rail
- ✅ **Cryptographic price verification** (trustless)
- ✅ Optional payer privacy (future: ZK-proof-of-payment)
- ✅ Anonymous rate limiting (future: RLN integration)

**Result**: x402 becomes **trustless** instead of trust-based. Agents can safely transact without manual audits.

---

## 🛣️ Roadmap

### ✅ Phase 1: MVP (Complete)

- [x] WASM pricing circuit
- [x] Rust prover with zkEngine
- [x] TypeScript x402 middleware
- [x] Example API with real proofs

### 🚧 Phase 2: Production (2-4 weeks)

- [ ] Standalone verifier binary (agents can verify proofs locally)
- [ ] Rust FFI or gRPC prover service (replace shell exec)
- [ ] On-chain tariff commitments (IPFS + smart contract)
- [ ] CI/CD and Docker deployment

### 🔮 Phase 3: Advanced Features (4-8 weeks)

- [ ] ZK-Proof-of-Payment (hide payer address)
- [ ] Agent authorization proofs (JOLT Atlas integration)
- [ ] Anonymous rate limiting (RLN)
- [ ] Compressed proofs (Groth16) for on-chain verification

---

## 📖 API Reference

### Middleware: `zkx402Middleware(config)`

**Config**:
```typescript
{
  tariff: PublicTariff,           // Public rate card
  computeMetadata: (req) => {     // Extract request metadata
    tokens: bigint,
    tier: 0 | 1 | 2,
  },
  facilitator: {
    chain: string,                // e.g., "base-sepolia"
    asset: string,                // e.g., "usdc"
  },
  prover?: ZKProver,              // Custom prover instance
}
```

**Behavior**:
1. If `X-PAYMENT` header is missing → Generate ZK proof + return 402
2. If `X-PAYMENT` exists → Verify payment + call next()

### Prover: `ZKProver`

```typescript
const prover = new ZKProver();

// Generate proof
const proof = await prover.generatePricingProof(
  { tokens: 1200n, tier: 1 },
  PUBLIC_TARIFF
);

// Verify proof
const valid = await prover.verifyPricingProof(proof);
```

---

## 🤝 Contributing

This is an open-source project! Contributions welcome.

**Areas for help**:
- Optimize zkEngine proving time
- Build standalone verifier
- Add more pricing models (volume discounts, credits, etc.)
- Integrate with real x402 facilitators (Coinbase CDP)

---

## 📜 License

MIT License - See LICENSE files in respective directories.

**zkEngine**: Apache-2.0 / MIT (see `zkEngine_dev/LICENSE-*`)

---

## 🙏 Acknowledgments

- [zkEngine](https://github.com/ICME-Lab/zkEngine_dev) by ICME Lab
- [Nebula](https://eprint.iacr.org/2024/1605) proving scheme
- [x402 Protocol](https://docs.cdp.coinbase.com/x402) by Coinbase
- Inspiration from [RLN](https://rate-limiting-nullifier.github.io/rln-docs) for anonymous rate limiting

---

## 📬 Contact

**Questions? Ideas?**

- Open an issue on GitHub
- Join the discussion on x402 Discord/forums
- Build with us!

---

**Built with ❤️ for the autonomous agent economy**

*"Don't trust, verify — even the price."* 🔐
