# ZK-Agent-Authorization: Proof-Carrying Authorization for AI Agents

**Cryptographic proofs that AI agents are authorized to make payments**

---

## 📖 What This Does (Plain English)

**Imagine your AI assistant can spend your money, but you want to make sure it never overspends or buys the wrong things.**

**ZK-Agent-Authorization solves this** by giving your AI a **digital mandate** that:
- ✅ Proves the AI is authorized to make this specific payment
- ✅ Shows the payment follows your budget rules
- ✅ Hides your private financial details (balance, exact limits)
- ✅ Creates an audit trail you can verify later

**Think of it like**: Giving your teenager a credit card with spending limits, but instead of trusting the bank to enforce limits, you get a mathematical proof for every purchase that shows "yes, this was within the rules."

---

## 🎯 The Problem

### **Scenario**: Enterprise with 100 AI Agents

**CompanyX deploys 100 AI agents** to:
- Research competitors
- Generate marketing content
- Analyze customer data
- Book cloud compute

**Each agent makes dozens of x402 payments daily.**

**The Problem**:
```
CFO: "We spent $50,000 on AI agents last month. Prove every payment
     was authorized and within policy."

IT Team: "Uh... we have logs?"

CFO: "Logs can be faked. I need cryptographic proof."
```

**Without ZK-Agent-Authorization**:
- ❌ No proof that payments followed policy
- ❌ Agents could overspend (bugs, hacks, drift)
- ❌ No audit trail for compliance (SOC2, GDPR)
- ❌ Must trust agent code + infrastructure

**With ZK-Agent-Authorization**:
- ✅ **Every payment has a ZK proof**: "This spend was authorized by the policy model"
- ✅ **Audit trail**: All proofs are stored and verifiable
- ✅ **Privacy**: Balance and limits stay hidden
- ✅ **Compliance**: Cryptographic non-repudiation

---

## 🏗️ How It Works

### **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│  Enterprise Policy (ONNX Model)                             │
│                                                              │
│  authorize(                                                  │
│    amount,              // How much?                         │
│    merchant_trust,      // Who?                              │
│    category,            // What for?                         │
│    spend_velocity,      // How fast are we spending?         │
│    current_balance      // Can we afford it?                 │
│  ) -> authorized: bool                                       │
│                                                              │
│  Example Rules:                                              │
│  - Single payment < $100                                     │
│  - Daily spend < $1,000                                      │
│  - Only trusted merchants (score > 0.8)                      │
│  - No "entertainment" category                               │
│  - Must have 2x balance                                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  AI Agent Makes Payment Request                             │
│                                                              │
│  "I want to pay $42 to OpenAI for LLM inference"            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  JOLT Atlas ZK Prover (ZKML)                                │
│                                                              │
│  Inputs (private):                                           │
│    amount: 42                                                │
│    merchant_trust: 0.95 (OpenAI = trusted)                  │
│    category: 1 (AI/ML)                                       │
│    spend_velocity: 450 (spent $450 today)                   │
│    current_balance: 5000                                     │
│                                                              │
│  Model: policy_v1.onnx (5→16→8→2)                           │
│                                                              │
│  Output (public):                                            │
│    authorized: true                                          │
│    confidence: 0.98                                          │
│                                                              │
│  ZK Proof: "I ran the policy model and got authorized=true" │
│  (Hides: balance, exact velocity, merchant score)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  x402 Payment + Authorization Proof                         │
│                                                              │
│  Headers:                                                    │
│    X-PAYMENT: <payment-token>                               │
│    X-AUTH-PROOF: <zk-proof>                                 │
│                                                              │
│  Body:                                                       │
│    {                                                         │
│      "amount": 42,                                           │
│      "authorization": {                                      │
│        "authorized": true,                                   │
│        "proof": "eyJwcm9vZi...",                             │
│        "policy_hash": "0xabc123...",                         │
│        "timestamp": "2025-01-26T..."                         │
│      }                                                       │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧪 Technical Details

### **JOLT Atlas ZKML**

**Why JOLT Atlas?**
- ✅ **Fastest ZKML**: 0.7s proving time (5-6x faster than competitors)
- ✅ **Tiny proofs**: 524 bytes (can attach to every payment)
- ✅ **ONNX compatible**: Use any ML framework (PyTorch, TensorFlow, scikit-learn)
- ✅ **Production-ready**: Already used in AgentKit ACP for authorization

**Performance**:
- **Proving time**: ~700ms for small policy models
- **Verification time**: <50ms
- **Proof size**: 524 bytes
- **Model size**: Up to ~1,000 params (5→16→8→2 layers)

### **Policy Model (ONNX)**

**Example policy model** (trained with scikit-learn):

```python
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
import numpy as np

# Training data: [amount, merchant_trust, category, velocity, balance] -> authorized
X_train = np.array([
    [10, 0.9, 1, 100, 5000],   # ✅ authorized
    [50, 0.95, 1, 200, 5000],  # ✅ authorized
    [200, 0.6, 2, 500, 5000],  # ❌ denied (low trust)
    [1000, 0.9, 1, 900, 5000], # ❌ denied (velocity too high)
    [50, 0.9, 3, 100, 100],    # ❌ denied (balance too low)
])
y_train = np.array([1, 1, 0, 0, 0])  # 1=authorized, 0=denied

# Train policy model
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
model.fit(X_train, y_train)

# Convert to ONNX
onnx_model = convert_sklearn(model, "policy_model",
    [("input", FloatTensorType([None, 5]))])

# Save for JOLT Atlas
with open("policy_v1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Model architecture**:
```
Input (5 features) → Dense(16) → ReLU → Dense(8) → ReLU → Dense(2) → Softmax
                      ↓                    ↓                  ↓
                   [authorized, denied]
```

### **Proof Generation Flow**

```rust
// JOLT Atlas proof generation (Rust)
use jolt_atlas::{JoltSNARK, ONNXModel};

// 1. Load policy model
let policy_model = ONNXModel::from_file("policy_v1.onnx")?;

// 2. Prepare inputs (from payment request)
let inputs = vec![
    42.0,      // amount
    0.95,      // merchant_trust
    1.0,       // category (AI/ML)
    450.0,     // spend_velocity
    5000.0,    // current_balance
];

// 3. Generate proof
let start = Instant::now();
let proof = JoltSNARK::prove(&policy_model, &inputs)?;
println!("Proof generated in {:?}", start.elapsed()); // ~700ms

// 4. Extract output
let authorized = proof.output[0] > proof.output[1]; // argmax
println!("Authorized: {}", authorized);

// 5. Serialize proof (524 bytes)
let proof_bytes = proof.serialize();
```

---

## 💰 Business Model

### **Why Enterprises Pay 10x More**

| Metric | ZK-Fair-Pricing | ZK-Agent-Authorization |
|--------|-----------------|------------------------|
| **Customer** | API sellers | Enterprises with AI agents |
| **Pain point** | Price transparency | Compliance/audit trails |
| **Willingness to pay** | Low-Medium ($0.001/proof) | **High** ($0.01-0.05/proof) |
| **Volume** | High (millions/month) | Medium (thousands/month) |
| **LTV** | $100-500/mo | **$5K-50K/year** |

### **Pricing Tiers**

| Plan | Price per Proof | Use Case | Target |
|------|----------------|----------|--------|
| **Startup** | $0.01 | <10 agents, dev/test | Small companies |
| **Business** | $0.005 | 10-100 agents | Mid-market |
| **Enterprise** | $0.002 | 100+ agents | Large enterprises |
| **White-label** | Custom | Embedded in MCP/ACP | Tool providers |

### **Revenue Model**

**Scenario**: 50-agent enterprise

- 50 agents × 20 payments/day = 1,000 payments/day
- 1,000 × 30 days = 30,000 proofs/month
- 30,000 × $0.005 = **$150/month**

**At scale** (100 enterprise customers):
- 100 × $150 = **$15,000/month** = **$180K ARR**

**Plus**:
- Setup fee: $500-1,000 per enterprise
- Custom policy models: $2,000-5,000 one-time
- White-label licensing: $25K/year to MCP server providers

**Total potential**: **$300K-500K ARR** in Year 1

---

## 🆚 vs. ZK-Fair-Pricing

### **Why Build Both?**

They're **complementary**, not competing:

| Dimension | ZK-Fair-Pricing | ZK-Agent-Authorization |
|-----------|-----------------|------------------------|
| **What it proves** | "Price is correct" | "Agent is authorized" |
| **Seller-side** | ✅ Yes | ❌ No |
| **Buyer-side** | ✅ Yes | ✅ **Yes** |
| **Who pays for it** | API sellers | Enterprises (agent owners) |
| **Proof tech** | zkEngine WASM | JOLT Atlas ONNX |
| **Proof time** | ~5-10s | ~0.7s |
| **Customer segment** | x402 API marketplace | Enterprise IT/Finance |
| **Revenue per customer** | $100-1,000/month | **$150-5,000/month** |

### **Combined Value**

**Full transaction with both proofs**:

```http
POST /api/llm/generate
X-PAYMENT: <payment-token>

Response (402):
  X-Pricing-Proof: <zkEngine proof>
    "This price ($0.42) was computed fairly"

  X-Auth-Proof: <JOLT proof>
    "This agent is authorized to spend $0.42"
```

**Buyer gets**: "The price is fair AND my agent is authorized"
**Seller gets**: "Payment verified AND buyer authorized"
**Both get**: **Complete trust layer**

---

## 🎯 Use Cases

### **1. Enterprise AI Agent Fleets**

**Scenario**: 100 agents for sales automation

- Agents research leads, draft emails, book meetings
- Each agent can spend up to $50/day
- Company policy: Only approved vendors, no personal expenses

**With ZK-Agent-Auth**:
- Every spend has a proof it followed policy
- CFO can audit all 100 agents in minutes
- Compliance team has cryptographic evidence

### **2. Autonomous Trading Bots**

**Scenario**: DeFi trading agent with $100K capital

- Agent makes 1,000 trades/day
- Must follow risk management rules
- Company needs audit trail for SEC

**With ZK-Agent-Auth**:
- Each trade has proof it followed risk model
- Regulators can verify compliance
- Trade secrets (strategy, positions) stay hidden

### **3. Multi-Agent Collaboration**

**Scenario**: Parent agent delegates to sub-agents

- Parent agent has $10K budget
- Delegates $1K to 10 sub-agents
- Each sub-agent must prove budget compliance

**With ZK-Agent-Auth**:
- Sub-agents prove to parent: "I'm within my $1K limit"
- Parent proves to company: "All subs are within budget"
- Hierarchical authorization with ZK proofs

### **4. Marketplace Agent Reputation**

**Scenario**: Agent marketplace (like x402 Bazaar for agents)

- Agents need reputation to get hired
- Reputation = "% of payments that were authorized"
- Agents prove authorization without revealing private data

**With ZK-Agent-Auth**:
- Agent: "I have 99% authorization rate (proof attached)"
- Employer: *Verifies proof* → "Hired!"
- Competitors don't learn agent's clients or budget

---

## 🛠️ Implementation Plan

### **Phase 1: MVP** (2-3 weeks)

- [ ] Clone JOLT Atlas (https://github.com/ICME-Lab/jolt-atlas)
- [ ] Train sample policy model (scikit-learn → ONNX)
- [ ] Build Rust prover service (JOLT wrapper)
- [ ] Create TypeScript client SDK
- [ ] Test with 5 design partner companies

**Deliverable**: Working proof generation + verification

### **Phase 2: x402 Integration** (2 weeks)

- [ ] Add `X-Auth-Proof` header to x402 middleware
- [ ] Build policy model management UI
- [ ] Create audit dashboard (view all proofs)
- [ ] Write integration docs

**Deliverable**: Complete x402 + ZK-Auth service

### **Phase 3: Production** (2-3 weeks)

- [ ] Horizontal scaling (multiple provers)
- [ ] Proof caching (avoid re-proving same request)
- [ ] SLA monitoring (99.9% uptime)
- [ ] Enterprise onboarding flow

**Deliverable**: Production-ready SaaS

**Total timeline**: 6-8 weeks to revenue

---

## 📊 Integration Example

### **Server-Side** (TypeScript + Rust)

```typescript
import { JoltAtlasProver } from './jolt-prover.js';

// Initialize prover with policy model
const prover = new JoltAtlasProver({
  modelPath: './models/policy_v1.onnx',
  policyHash: '0xabc123...',
});

// x402 middleware with agent authorization
app.post('/api/llm/generate',
  zkx402Middleware({ /* ... */ }),
  async (req, res) => {
    // Extract agent metadata
    const agentContext = {
      amount: req.zkx402.price,
      merchantTrust: 0.95,
      category: getCategoryId(req.path),
      spendVelocity: await getAgentVelocity(req.agentId),
      currentBalance: await getAgentBalance(req.agentId),
    };

    // Generate authorization proof
    const authProof = await prover.prove(agentContext);

    if (!authProof.authorized) {
      return res.status(403).json({
        error: 'Agent not authorized to make this payment',
        reason: authProof.reason,
      });
    }

    // Attach proof to response
    res.setHeader('X-Auth-Proof', JSON.stringify({
      proof: authProof.proof,
      authorized: true,
      policyHash: prover.policyHash,
    }));

    // Process request...
    res.json({ result: '...' });
  }
);
```

### **Agent-Side** (TypeScript SDK)

```typescript
import { ZKAgentAuthClient } from '@zkx402/agent-auth';

// Initialize agent with policy model
const agent = new ZKAgentAuthClient({
  policyModel: './my-policy.onnx',
  balance: 5000,
  dailyLimit: 1000,
});

// Make request with authorization proof
const response = await agent.request('POST', '/api/llm/generate', {
  prompt: 'Hello world',
});

// Agent SDK automatically:
// 1. Checks if request is authorized per policy
// 2. Generates ZK proof if authorized
// 3. Attaches X-Auth-Proof header
// 4. Stores proof for audit trail

console.log('Authorized:', response.authorized);
console.log('Proof:', response.authProof);
```

---

## 🔐 Security Considerations

### **Policy Model Integrity**

**Problem**: What if someone swaps the policy model?

**Solution**: Hash commitment
```typescript
const policyHash = sha256(onnx_model_bytes);
// Publish hash on-chain or in company registry
// Every proof includes policy_hash
// Auditors verify: "Was this the approved policy?"
```

### **Proof Replay Attacks**

**Problem**: Reuse old proof for new payment

**Solution**: Include nonce + timestamp
```typescript
const proof = await prover.prove({
  ...agentContext,
  nonce: crypto.randomBytes(32),
  timestamp: Date.now(),
});
```

### **Model Extraction**

**Problem**: Can adversary extract policy from proofs?

**Answer**: No! ZK proofs hide model weights
- Proof reveals: "authorized = true"
- Proof hides: Why it was authorized, what the rules are

---

## 📈 Go-to-Market

### **Target Customers** (Year 1)

1. **Enterprises with AI agents** (50-500 employees)
   - Current: Manual approval for agent spends
   - Pain: No audit trail, risk of overspend
   - Willingness to pay: High ($5K-50K/year)

2. **MCP server providers**
   - Embed authorization in MCP tools
   - White-label licensing: $25K/year

3. **x402 marketplaces**
   - "ZK-Authorized Agent" badge
   - Agents with proofs get more jobs

### **Marketing Strategy**

**Month 1-2**: Build credibility
- Publish technical blog on JOLT Atlas + x402
- Demo at x402 conferences
- Case study with 1 design partner

**Month 3-4**: Direct sales
- Pitch to 20 enterprises (AI-heavy companies)
- Target: 5 paying customers by Month 4

**Month 5-6**: Scale
- Launch self-serve tier ($500/month)
- Partner with MCP ecosystem
- Target: 10 paying customers

---

## 🎯 Success Metrics

**Technical**:
- ✅ <1s proof generation (JOLT Atlas achieves this)
- ✅ 99.9% uptime
- ✅ <100ms verification time

**Business**:
- Month 3: 5 design partners
- Month 6: 10 paying customers
- Month 12: $150K ARR
- Year 2: $500K ARR

---

## 🚀 Next Steps

### **Want to Build This?**

1. **Read JOLT Atlas docs**: https://github.com/ICME-Lab/jolt-atlas
2. **Clone the repo**: `git clone https://github.com/ICME-Lab/jolt-atlas`
3. **Train a sample policy model**: See `policy_model_example.py` above
4. **Generate your first proof**: `cargo run --example acp` in JOLT Atlas
5. **Integrate with zkx402**: Add `X-Auth-Proof` header to middleware

### **Resources**

- JOLT Atlas: https://github.com/ICME-Lab/jolt-atlas
- AgentKit ACP example: Real production usage of JOLT for authorization
- x402 protocol: https://docs.cdp.coinbase.com/x402
- ONNX: https://onnx.ai/

---

**Ready to build?** This is the perfect complement to ZK-Fair-Pricing! 🚀🔐
