/**
 * Combined ZKx402 Middleware - Fair-Pricing + Agent-Authorization
 *
 * This middleware supports BOTH ZK proof types:
 * 1. Fair-Pricing: Seller proves price is correct (X-Pricing-Proof header)
 * 2. Agent-Auth: Buyer proves agent is authorized (X-Agent-Auth-Proof header)
 *
 * Full flow:
 * 1. Agent makes request without payment
 * 2. Server returns 402 with Fair-Pricing proof
 * 3. Agent verifies pricing proof
 * 4. Agent generates Agent-Auth proof
 * 5. Agent sends payment + Agent-Auth proof
 * 6. Server verifies both proofs and processes request
 */

import { Request, Response, NextFunction } from "express";
import type {
  PublicTariff,
  PaymentRequirement,
  XPaymentHeader,
  FacilitatorConfig,
  PaymentScheme,
  Network,
} from "./x402-types.js";
import { ZKProver } from "./zk-prover.js";
import {
  encodeXPaymentHeader,
  decodeXPaymentHeader,
  hashTariff,
} from "./x402-utils.js";

export interface CombinedZKx402Config {
  // Pricing configuration (seller-side)
  tariff: PublicTariff;
  computeMetadata: (req: Request) => { tokens: bigint; tier: number };
  facilitator: FacilitatorConfig;
  prover?: ZKProver;

  // Authorization configuration (buyer-side)
  requireAgentAuth?: boolean; // Whether to require agent authorization proofs
  authServiceUrl?: string; // URL of hybrid auth router for verification
}

/**
 * Combined middleware that generates Fair-Pricing proofs
 * and optionally requires Agent-Authorization proofs
 */
export function zkx402CombinedMiddleware(config: CombinedZKx402Config) {
  const prover = config.prover || new ZKProver();
  const tariffHash = hashTariff(config.tariff);

  return async (req: Request, res: Response, next: NextFunction) => {
    // =========================================================================
    // DISCOVERY: OPTIONS pre-flight
    // =========================================================================
    if (req.method === "OPTIONS") {
      return handleCombinedDiscovery(req, res, config, tariffHash);
    }

    // =========================================================================
    // STEP 1: Check if payment + auth headers are present
    // =========================================================================
    const paymentHeader = req.headers["x-payment"] as string | undefined;
    const agentAuthProofHeader = req.headers["x-agent-auth-proof"] as
      | string
      | undefined;

    if (!paymentHeader) {
      // No payment: Generate Fair-Pricing proof and return 402
      return await handlePricingChallenge(req, res, config, prover, tariffHash);
    }

    // =========================================================================
    // STEP 2: Payment present, verify it
    // =========================================================================
    try {
      const payment = decodeXPaymentHeader(paymentHeader);

      // Verify payment with facilitator
      const facilitatorValid = await verifyWithFacilitator(
        payment,
        config.facilitator
      );

      if (!facilitatorValid) {
        return res.status(402).json({
          error: "Payment verification failed",
          details: "Payment token is invalid or expired",
        });
      }

      // =========================================================================
      // STEP 3: Check agent authorization proof (if required)
      // =========================================================================
      if (config.requireAgentAuth) {
        if (!agentAuthProofHeader) {
          return res.status(403).json({
            error: "Agent authorization required",
            details: {
              message:
                "This API requires a zero-knowledge proof that the agent is authorized to spend",
              header: "X-Agent-Auth-Proof",
              proofFormat: {
                type: "jolt" | "zkengine",
                proofData: "base64-encoded proof",
                publicInputs: {
                  transactionAmount: "micro-USDC",
                  vendorId: "vendor identifier",
                  timestamp: "Unix timestamp",
                },
              },
              helpUrl: config.authServiceUrl || "http://localhost:3403/info",
            },
          });
        }

        // Verify agent authorization proof
        try {
          const authProof = JSON.parse(agentAuthProofHeader);
          const authValid = await verifyAgentAuthProof(authProof, config);

          if (!authValid || !authProof.authorized) {
            return res.status(403).json({
              error: "Agent not authorized",
              details:
                "Zero-knowledge proof confirms agent violates spending policy",
            });
          }

          // Store auth proof data for logging/audit
          (req as any).agentAuthProof = authProof;
        } catch (authError) {
          return res.status(400).json({
            error: "Invalid agent authorization proof",
            message:
              authError instanceof Error ? authError.message : String(authError),
          });
        }
      }

      // =========================================================================
      // STEP 4: All checks passed, settle payment and proceed
      // =========================================================================
      await settlementWithFacilitator(payment, config.facilitator);

      // Store payment data for downstream handlers
      (req as any).x402Payment = payment;

      // Log successful transaction with both proofs
      console.log(
        `✅ [ZKx402 Combined] Transaction authorized:\n` +
          `   Amount: ${payment.payload.amount} ${config.facilitator.asset}\n` +
          `   Fair-Pricing: ✓ (server proved correct pricing)\n` +
          `   Agent-Auth: ${config.requireAgentAuth ? "✓ (agent proved authorization)" : "N/A"}`
      );

      next();
    } catch (error) {
      console.error("[ZKx402 Combined] Error:", error);
      res.status(500).json({
        error: "Payment processing failed",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  };
}

/**
 * Handle discovery with combined capabilities
 */
function handleCombinedDiscovery(
  req: Request,
  res: Response,
  config: CombinedZKx402Config,
  tariffHash: string
) {
  // Set x402 discovery headers
  res.setHeader("X-Accepts-Payment", `${config.facilitator.chain}:${config.facilitator.asset}:*`);
  res.setHeader("Access-Control-Expose-Headers", "X-Accepts-Payment, X-ZK-Pricing-Enabled, X-ZK-Agent-Auth-Required");

  // ZK-specific headers
  res.setHeader("X-ZK-Pricing-Enabled", "true");
  res.setHeader("X-ZK-Proof-Type", "zkengine-wasm");

  if (config.requireAgentAuth) {
    res.setHeader("X-ZK-Agent-Auth-Required", "true");
    res.setHeader(
      "X-ZK-Agent-Auth-Service",
      config.authServiceUrl || "http://localhost:3403"
    );
  }

  // Return tariff in body
  res.json({
    x402: {
      scheme: "exact" as const,
      network: config.facilitator.chain as Network,
      asset: config.facilitator.asset,
    },
    zkPricing: {
      enabled: true,
      tariff: config.tariff,
      tariffHash,
      proofType: "zkengine-wasm",
    },
    zkAgentAuth: config.requireAgentAuth
      ? {
          required: true,
          serviceUrl: config.authServiceUrl || "http://localhost:3403",
          supportedBackends: ["jolt", "zkengine"],
        }
      : {
          required: false,
        },
  });
}

/**
 * Handle 402 response with Fair-Pricing proof
 */
async function handlePricingChallenge(
  req: Request,
  res: Response,
  config: CombinedZKx402Config,
  prover: ZKProver,
  tariffHash: string
) {
  const metadata = config.computeMetadata(req);
  const priceInMicroUnits = BigInt(
    config.tariff.tiers[metadata.tier].base_price +
      Number(metadata.tokens) * config.tariff.tiers[metadata.tier].per_unit_price
  );

  // Generate ZK proof of correct pricing
  const zkProof = await prover.generatePricingProof(metadata, config.tariff);

  // Build payment requirement
  const paymentReq: PaymentRequirement = {
    scheme: "exact" as PaymentScheme,
    network: config.facilitator.chain as Network,
    maxAmountRequired: priceInMicroUnits.toString(),
    resource: req.path,
    description: `${metadata.tokens} tokens at tier ${metadata.tier}`,
    mimeType: "application/json",
    payTo: config.facilitator.receiverAddress || "0x...",
    maxTimeoutSeconds: 300,
    asset: config.facilitator.asset,
    extra: {
      zkProof,
      tariffHash,
      requiresAgentAuth: config.requireAgentAuth || false,
    },
  };

  // Set headers
  res.setHeader(
    "X-Accept-Payment",
    `${config.facilitator.chain}:${config.facilitator.asset}:${priceInMicroUnits}`
  );
  res.setHeader("X-Pricing-Proof", JSON.stringify(zkProof));

  if (config.requireAgentAuth) {
    res.setHeader("X-Agent-Auth-Required", "true");
  }

  res.status(402).json({
    error: "Payment Required",
    details: paymentReq,
  });
}

/**
 * Verify agent authorization proof
 */
async function verifyAgentAuthProof(
  proof: any,
  config: CombinedZKx402Config
): Promise<boolean> {
  // Call auth service to verify proof
  const authServiceUrl =
    config.authServiceUrl || "http://localhost:3403/verify";

  try {
    const response = await fetch(authServiceUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ proof: proof.proof }),
    });

    if (!response.ok) {
      return false;
    }

    const result = await response.json();
    return result.valid === true;
  } catch (error) {
    console.error("[Agent Auth] Verification failed:", error);
    return false;
  }
}

/**
 * Verify payment with facilitator (placeholder)
 */
async function verifyWithFacilitator(
  payment: XPaymentHeader,
  facilitator: FacilitatorConfig
): Promise<boolean> {
  // Real implementation would call facilitator's verification endpoint
  // For now, just check structure
  return payment.scheme === "exact" && !!payment.payload;
}

/**
 * Settle payment with facilitator (placeholder)
 */
async function settlementWithFacilitator(
  payment: XPaymentHeader,
  facilitator: FacilitatorConfig
): Promise<void> {
  // Real implementation would call facilitator's settlement endpoint
  console.log(`[Settlement] Processing payment via ${facilitator.chain}`);
}
