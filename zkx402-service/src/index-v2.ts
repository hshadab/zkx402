/**
 * ZKx402 Fair-Pricing Service (Production)
 *
 * Spec-compliant x402 implementation with zero-knowledge pricing proofs
 */

import express from "express";
import { zkx402Middleware } from "./x402-middleware-v2.js";
import type { PublicTariff } from "./types.js";

const app = express();
app.use(express.json());

// ==============================================================================
// CONFIGURATION
// ==============================================================================

const PUBLIC_TARIFF: PublicTariff = {
  tiers: {
    basic: {
      basePrice: 10_000n, // $0.01
      perUnitPrice: 100n, // $0.0001 per token
    },
    pro: {
      basePrice: 50_000n, // $0.05
      perUnitPrice: 350n, // $0.00035 per token
    },
    enterprise: {
      basePrice: 100_000n, // $0.10
      perUnitPrice: 800n, // $0.0008 per token
    },
  },
  multiplier: 10_000, // 1.0x (no surge pricing)
};

const SERVICE_CONFIG = {
  name: "ZKx402 LLM API",
  description: "AI inference with cryptographically verified fair pricing",
  baseUrl: process.env.BASE_URL || "http://localhost:3402",
  recipientAddress:
    process.env.RECIPIENT_ADDRESS || "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb1",
  network: (process.env.NETWORK as any) || "base-sepolia",
  assetAddress:
    process.env.ASSET_ADDRESS ||
    "0x036CbD53842c5426634e7929541eC2318f3dCF7e", // USDC on Base Sepolia
  facilitatorVerifyUrl:
    process.env.FACILITATOR_VERIFY_URL ||
    "https://api.cdp.coinbase.com/x402/verify",
  facilitatorSettleUrl:
    process.env.FACILITATOR_SETTLE_URL ||
    "https://api.cdp.coinbase.com/x402/settle",
};

// ==============================================================================
// ROUTES
// ==============================================================================

/**
 * GET /tariff - Public tariff endpoint (for agent discovery)
 */
app.get("/tariff", (req, res) => {
  res.json({
    tariff: {
      basic: {
        basePrice: PUBLIC_TARIFF.tiers.basic.basePrice.toString(),
        perUnitPrice: PUBLIC_TARIFF.tiers.basic.perUnitPrice.toString(),
      },
      pro: {
        basePrice: PUBLIC_TARIFF.tiers.pro.basePrice.toString(),
        perUnitPrice: PUBLIC_TARIFF.tiers.pro.perUnitPrice.toString(),
      },
      enterprise: {
        basePrice: PUBLIC_TARIFF.tiers.enterprise.basePrice.toString(),
        perUnitPrice:
          PUBLIC_TARIFF.tiers.enterprise.perUnitPrice.toString(),
      },
      multiplier: PUBLIC_TARIFF.multiplier,
    },
    hash: require("crypto")
      .createHash("sha256")
      .update(JSON.stringify(PUBLIC_TARIFF))
      .digest("hex"),
    description: "All prices in micro-dollars (1,000,000 = $1.00)",
  });
});

/**
 * POST /api/llm/generate - LLM API with ZK-Fair-Pricing
 */
app.post(
  "/api/llm/generate",
  zkx402Middleware({
    tariff: PUBLIC_TARIFF,
    computeMetadata: (req) => {
      const prompt = req.body.prompt || "";
      const estimatedTokens = Math.ceil(prompt.length / 4);
      const tier = (req.body.tier as 0 | 1 | 2) || 0;

      return {
        tokens: BigInt(estimatedTokens),
        tier,
      };
    },
    facilitator: {
      verifyUrl: SERVICE_CONFIG.facilitatorVerifyUrl,
      settleUrl: SERVICE_CONFIG.facilitatorSettleUrl,
    },
    payment: {
      network: SERVICE_CONFIG.network,
      assetAddress: SERVICE_CONFIG.assetAddress,
      recipientAddress: SERVICE_CONFIG.recipientAddress,
      timeoutSeconds: 300, // 5 minutes
    },
    service: {
      name: SERVICE_CONFIG.name,
      description: SERVICE_CONFIG.description,
      baseUrl: SERVICE_CONFIG.baseUrl,
    },
  }),
  async (req, res) => {
    // This only runs if payment was verified
    const prompt = req.body.prompt;
    const x402Context = (req as any).x402;

    res.json({
      result: `Mock LLM response to: "${prompt}"`,
      usage: {
        promptTokens: Math.ceil(prompt.length / 4),
        completionTokens: 50,
        totalTokens: Math.ceil(prompt.length / 4) + 50,
      },
      x402: {
        paid: true,
        zkVerified: x402Context.zkVerified,
        transactionHash: x402Context.settlement.transactionHash,
        network: SERVICE_CONFIG.network,
      },
    });
  }
);

/**
 * POST /api/compute - Compute API with tiered pricing
 */
app.post(
  "/api/compute",
  zkx402Middleware({
    tariff: PUBLIC_TARIFF,
    computeMetadata: (req) => ({
      tokens: BigInt(req.body.computeUnits || 100),
      tier: (req.body.tier as 0 | 1 | 2) || 1,
    }),
    facilitator: {
      verifyUrl: SERVICE_CONFIG.facilitatorVerifyUrl,
      settleUrl: SERVICE_CONFIG.facilitatorSettleUrl,
    },
    payment: {
      network: SERVICE_CONFIG.network,
      assetAddress: SERVICE_CONFIG.assetAddress,
      recipientAddress: SERVICE_CONFIG.recipientAddress,
      timeoutSeconds: 300,
    },
    service: {
      name: "ZKx402 Compute API",
      description: "Compute services with ZK-verified pricing",
      baseUrl: SERVICE_CONFIG.baseUrl,
    },
  }),
  async (req, res) => {
    const units = req.body.computeUnits || 100;
    const x402Context = (req as any).x402;

    res.json({
      result: `Computed ${units} units`,
      x402: {
        paid: true,
        zkVerified: x402Context.zkVerified,
      },
    });
  }
);

/**
 * GET /health - Health check
 */
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    service: "zkx402-fair-pricing",
    version: "1.0.0",
    features: {
      x402: true,
      zkPricing: true,
      proofType: "zkengine-wasm",
      network: SERVICE_CONFIG.network,
    },
  });
});

/**
 * GET /.well-known/x402 - x402 discovery endpoint (proposed standard)
 */
app.get("/.well-known/x402", (req, res) => {
  res.json({
    x402Version: 1,
    service: {
      name: SERVICE_CONFIG.name,
      description: SERVICE_CONFIG.description,
      url: SERVICE_CONFIG.baseUrl,
    },
    payment: {
      schemes: ["zkproof", "exact"],
      networks: [SERVICE_CONFIG.network],
      assets: [
        {
          network: SERVICE_CONFIG.network,
          address: SERVICE_CONFIG.assetAddress,
          symbol: "USDC",
          decimals: 6,
        },
      ],
    },
    zkPricing: {
      enabled: true,
      proofType: "zkengine-wasm",
      tariffHash: require("crypto")
        .createHash("sha256")
        .update(JSON.stringify(PUBLIC_TARIFF))
        .digest("hex"),
      tariffEndpoint: `${SERVICE_CONFIG.baseUrl}/tariff`,
    },
    endpoints: [
      {
        path: "/api/llm/generate",
        method: "POST",
        description: "LLM text generation",
        pricing: "dynamic",
        schema: {
          input: { prompt: "string", tier: "0 | 1 | 2" },
          output: { result: "string", usage: "object" },
        },
      },
      {
        path: "/api/compute",
        method: "POST",
        description: "Compute services",
        pricing: "dynamic",
        schema: {
          input: { computeUnits: "number", tier: "0 | 1 | 2" },
          output: { result: "string" },
        },
      },
    ],
  });
});

// ==============================================================================
// START SERVER
// ==============================================================================

const PORT = process.env.PORT || 3402;

app.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔐 ZKx402 Fair-Pricing Service (Production)                ║
║                                                               ║
║   x402-compliant with Zero-Knowledge Pricing Proofs          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Server:        http://localhost:${PORT}
Network:       ${SERVICE_CONFIG.network}
Asset:         USDC (${SERVICE_CONFIG.assetAddress})

Discovery Endpoints:
  GET  /.well-known/x402  - Service discovery (JSON)
  GET  /tariff            - Public pricing tariff
  GET  /health            - Health check

Payment Endpoints (x402):
  POST /api/llm/generate  - LLM API (ZK-verified pricing)
  POST /api/compute       - Compute API (ZK-verified pricing)

Features:
  ✅ x402 protocol v1 (spec-compliant)
  ✅ ZK-Fair-Pricing proofs (zkEngine)
  ✅ Agent discovery (OPTIONS pre-flight)
  ✅ Client-side verification (X-Pricing-Proof header)

Try Discovery:
  curl -X OPTIONS http://localhost:${PORT}/api/llm/generate

Try Payment Flow:
  curl -X POST http://localhost:${PORT}/api/llm/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompt":"Hello","tier":1}'

  → Returns 402 with ZK pricing proof
  → Agents verify proof matches public tariff
  → Agents pay and retry with X-Payment header
`);
});

export { app };
