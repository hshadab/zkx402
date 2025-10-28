import express, { Request, Response } from "express";
import cors from "cors";
import { PolicyClassifier } from "./classifier.js";
import { JoltClient } from "./jolt-client.js";
import { ZkEngineClient } from "./zkengine-client.js";
import type { AuthorizationRequest, AuthorizationResponse } from "./types.js";

const app = express();
const PORT = process.env.PORT || 3403;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize prover clients
const joltClient = new JoltClient();
const zkEngineClient = new ZkEngineClient();

/**
 * Health check endpoint
 */
app.get("/health", (req: Request, res: Response) => {
  res.json({
    status: "healthy",
    service: "zkx402-hybrid-auth-router",
    backends: {
      jolt: "available",
      zkengine: "available",
    },
  });
});

/**
 * Get service information
 */
app.get("/info", (req: Request, res: Response) => {
  res.json({
    name: "ZKx402 Hybrid Authorization Router",
    version: "0.1.0",
    description:
      "Routes authorization requests to JOLT zkVM or zkEngine WASM based on policy complexity",
    backends: [
      {
        name: "JOLT zkVM",
        type: "simple",
        provingTime: "~0.7s",
        proofSize: "524 bytes",
        features: [
          "Velocity checks",
          "Numeric comparisons",
          "Trust scoring",
        ],
      },
      {
        name: "zkEngine WASM",
        type: "complex",
        provingTime: "~5-10s",
        proofSize: "~1-2KB",
        features: [
          "Whitelist checking",
          "Business hours validation",
          "Budget tracking",
          "Complex boolean logic",
        ],
      },
    ],
  });
});

/**
 * Classify a policy (without generating proof)
 */
app.post("/classify", (req: Request, res: Response) => {
  try {
    const request = req.body as AuthorizationRequest;
    const classification = PolicyClassifier.classify(request);

    res.json({
      classification: {
        policyType: classification.policyType,
        backend: classification.backend,
        reason: classification.reason,
        description: PolicyClassifier.describe(classification),
      },
      estimatedProvingTimeMs: PolicyClassifier.estimateProvingTime(classification),
      estimatedProofSizeBytes: PolicyClassifier.estimateProofSize(classification),
    });
  } catch (error) {
    res.status(400).json({
      error: "Classification failed",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

/**
 * Generate authorization proof (main endpoint)
 */
app.post("/authorize", async (req: Request, res: Response) => {
  const startTime = Date.now();

  try {
    const request = req.body as AuthorizationRequest;

    // Validate request
    if (!request.transactionAmount || !request.balance) {
      return res.status(400).json({
        error: "Invalid request",
        message: "transactionAmount and balance are required",
      });
    }

    // 1. Classify the policy
    const classification = PolicyClassifier.classify(request);

    console.log(
      `[Authorize] ${classification.backend.toUpperCase()} - ${classification.reason}`
    );

    // 2. Route to appropriate prover
    let response: AuthorizationResponse;

    if (classification.backend === "jolt") {
      response = await joltClient.generateProof(request);
    } else {
      response = await zkEngineClient.generateProof(request);
    }

    // 3. Return response with routing metadata
    const totalTime = Date.now() - startTime;

    res.json({
      ...response,
      routing: {
        policyType: classification.policyType,
        backend: classification.backend,
        reason: classification.reason,
        totalTimeMs: totalTime,
      },
    });
  } catch (error) {
    console.error("[Authorize] Error:", error);
    res.status(500).json({
      error: "Proof generation failed",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

/**
 * Verify an authorization proof
 */
app.post("/verify", async (req: Request, res: Response) => {
  try {
    const { proof } = req.body;

    if (!proof || !proof.type) {
      return res.status(400).json({
        error: "Invalid request",
        message: "proof with type is required",
      });
    }

    let valid: boolean;

    if (proof.type === "jolt") {
      valid = await joltClient.verifyProof(proof);
    } else if (proof.type === "zkengine") {
      valid = await zkEngineClient.verifyProof(proof);
    } else {
      return res.status(400).json({
        error: "Invalid proof type",
        message: `Unknown proof type: ${proof.type}`,
      });
    }

    res.json({
      valid,
      proofType: proof.type,
      verifiedAt: new Date().toISOString(),
    });
  } catch (error) {
    res.status(500).json({
      error: "Verification failed",
      message: error instanceof Error ? error.message : String(error),
    });
  }
});

/**
 * Get sample requests for testing
 */
app.get("/samples", (req: Request, res: Response) => {
  res.json({
    simple: {
      description: "Simple velocity check (uses JOLT)",
      request: {
        transactionAmount: "50000", // $0.05
        vendorId: "12345",
        timestamp: 1704117600,
        balance: "10000000", // $10.00
        velocityData: {
          velocity1h: "20000",
          velocity24h: "100000",
          vendorTrustScore: 80,
        },
        policyType: "simple" as const,
        policyParams: {
          maxSingleTxPercent: 10,
          maxVelocity1hPercent: 5,
          maxVelocity24hPercent: 20,
          minVendorTrust: 50,
        },
      },
    },
    complex: {
      description: "Complex policy with whitelist and business hours (uses zkEngine)",
      request: {
        transactionAmount: "500000", // $0.50
        vendorId: "5",
        timestamp: 1704117600, // Wed, business hours
        balance: "10000000",
        budgetData: {
          dailyBudgetRemaining: "2000000", // $2.00
        },
        whitelistData: {
          whitelistBitmap: "0x20", // Bit 5 set
        },
        policyType: "complex" as const,
        policyParams: {
          requireBusinessHours: true,
          requireWhitelist: true,
          requireDailyBudget: true,
        },
      },
    },
  });
});

// Start server
app.listen(PORT, () => {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("ZKx402 Hybrid Authorization Router");
  console.log("═══════════════════════════════════════════════════════════════");
  console.log(`\n✓ Server running on http://localhost:${PORT}`);
  console.log("\nEndpoints:");
  console.log(`  GET  /health        - Health check`);
  console.log(`  GET  /info          - Service information`);
  console.log(`  POST /classify      - Classify policy (no proof)`);
  console.log(`  POST /authorize     - Generate authorization proof`);
  console.log(`  POST /verify        - Verify authorization proof`);
  console.log(`  GET  /samples       - Get sample requests`);
  console.log("\nBackends:");
  console.log(`  • JOLT zkVM         - Simple policies (~0.7s, 524 bytes)`);
  console.log(`  • zkEngine WASM     - Complex policies (~5-10s, ~1-2KB)`);
  console.log("\n═══════════════════════════════════════════════════════════════\n");
});

export { app };
