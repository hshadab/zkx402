/**
 * ZKx402 Fair-Pricing Service
 *
 * Demo x402 API with cryptographic proof of fair pricing
 */

import express from "express";
import { zkx402Middleware } from "./middleware.js";
import type { PublicTariff } from "./types.js";

const app = express();
app.use(express.json());

// ==============================================================================
// PUBLIC TARIFF (known to all agents)
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

// ==============================================================================
// ROUTES
// ==============================================================================

/**
 * GET /tariff - Return public pricing tariff
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
        perUnitPrice: PUBLIC_TARIFF.tiers.enterprise.perUnitPrice.toString(),
      },
      multiplier: PUBLIC_TARIFF.multiplier,
    },
    description: "All prices in micro-dollars (1,000,000 = $1.00)",
  });
});

/**
 * POST /api/llm/generate - Example LLM API with ZK-Fair-Pricing
 *
 * First call: 402 Payment Required + ZK proof
 * Second call (with X-PAYMENT): 200 OK + response
 */
app.post(
  "/api/llm/generate",
  zkx402Middleware({
    tariff: PUBLIC_TARIFF,
    computeMetadata: (req) => {
      const prompt = req.body.prompt || "";
      const estimatedTokens = Math.ceil(prompt.length / 4); // Rough estimate
      const tier = (req.body.tier as 0 | 1 | 2) || 0;

      return {
        tokens: BigInt(estimatedTokens),
        tier,
      };
    },
    facilitator: {
      chain: "base-sepolia",
      asset: "usdc",
    },
  }),
  async (req, res) => {
    // This only runs if payment was provided and verified
    const prompt = req.body.prompt;

    res.json({
      result: `Mock LLM response to: "${prompt}"`,
      usage: {
        promptTokens: Math.ceil(prompt.length / 4),
        completionTokens: 50,
        totalTokens: Math.ceil(prompt.length / 4) + 50,
      },
      zkx402: {
        message: "Payment verified. This response was paid for with x402 + ZK proof.",
      },
    });
  }
);

/**
 * POST /api/compute - Example compute API with tiered pricing
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
      chain: "base-sepolia",
      asset: "usdc",
    },
  }),
  async (req, res) => {
    const units = req.body.computeUnits || 100;

    res.json({
      result: `Computed ${units} units`,
      zkx402: {
        paid: true,
      },
    });
  }
);

/**
 * Health check
 */
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    service: "zkx402-fair-pricing",
    version: "0.1.0",
    zkEngine: "available",
  });
});

// ==============================================================================
// START SERVER
// ==============================================================================

const PORT = process.env.PORT || 3402; // 3402 for x402 ;)

app.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🔐 ZKx402 Fair-Pricing Service                             ║
║                                                               ║
║   Zero-Knowledge Proofs for x402 Protocol                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

Server running on: http://localhost:${PORT}

Public Tariff:
  Basic:      $0.0100 base + $0.000100/token
  Pro:        $0.0500 base + $0.000350/token
  Enterprise: $0.1000 base + $0.000800/token

Endpoints:
  GET  /health            - Health check
  GET  /tariff            - View public tariff
  POST /api/llm/generate  - LLM API (with ZK proofs)
  POST /api/compute       - Compute API (with ZK proofs)

Example Usage:
  curl -X POST http://localhost:${PORT}/api/llm/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompt":"Hello world","tier":1}'

  This will return a 402 challenge with a ZK proof that
  the price was computed according to the public tariff.

  The X-Pricing-Proof header contains a zero-knowledge proof
  generated by zkEngine that any agent can verify!
`);
});

export { app };
