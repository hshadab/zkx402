/**
 * Combined Demo - Fair-Pricing + Agent-Authorization
 *
 * This demo shows the complete flow:
 * 1. Agent discovers API capabilities
 * 2. Agent makes request (gets 402 + Fair-Pricing proof)
 * 3. Agent verifies Fair-Pricing proof
 * 4. Agent generates Agent-Authorization proof
 * 5. Agent sends payment + Agent-Auth proof
 * 6. Server verifies both proofs and processes request
 */

import { ZKx402AgentCombined, type AuthPolicyConfig } from "../src/agent-sdk-combined.js";

const API_URL = "http://localhost:3402/api/llm/generate";
const AUTH_SERVICE_URL = "http://localhost:3403";

async function runDemo() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("ZKx402 Combined Demo - Fair-Pricing + Agent-Authorization");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const agent = new ZKx402AgentCombined(AUTH_SERVICE_URL);

  // =========================================================================
  // STEP 1: Discovery
  // =========================================================================
  console.log("📡 STEP 1: Discovering API capabilities...\n");

  const discovery = await agent.discover(API_URL);

  console.log("Discovery results:");
  console.log(`  x402 support:         ${discovery.supportsX402 ? "✓" : "✗"}`);
  console.log(`  ZK Pricing:           ${discovery.supportsZKPricing ? "✓" : "✗"}`);
  console.log(`  ZK Agent Auth:        ${discovery.supportsZKAgentAuth ? "✓" : "✗"}`);
  console.log(`  Auth required:        ${discovery.zkAgentAuthRequired ? "YES" : "NO"}`);

  if (discovery.tariff) {
    console.log("\nPublic Tariff:");
    discovery.tariff.tiers.forEach((tier, idx) => {
      const baseUsd = Number(tier.base_price) / 1_000_000;
      const perTokenUsd = Number(tier.per_unit_price) / 1_000_000;
      console.log(
        `  ${tier.name}: $${baseUsd.toFixed(2)} + $${perTokenUsd.toFixed(6)}/token`
      );
    });
  }

  console.log("\n───────────────────────────────────────────────────────────────\n");

  // =========================================================================
  // STEP 2: Configure Agent Policy
  // =========================================================================
  console.log("🔐 STEP 2: Configuring agent spending policy...\n");

  const policyConfig: AuthPolicyConfig = {
    // Private inputs (NEVER sent to server)
    balance: "10000000", // $10.00

    velocityData: {
      velocity1h: "200000", // $0.20 spent in last hour
      velocity24h: "500000", // $0.50 spent in last 24h
      vendorTrustScore: 80, // Trust score: 80/100
    },

    // Policy parameters
    policyParams: {
      maxSingleTxPercent: 10, // Max 10% of balance per transaction
      maxVelocity1hPercent: 5, // Max 5% of balance per hour
      maxVelocity24hPercent: 20, // Max 20% of balance per day
      minVendorTrust: 50, // Minimum vendor trust: 50/100
    },
  };

  console.log("Policy configured:");
  console.log(`  Balance:              $${Number(policyConfig.balance) / 1_000_000} (PRIVATE)`);
  console.log(`  Max per transaction:  ${policyConfig.policyParams?.maxSingleTxPercent}% of balance`);
  console.log(`  Max per hour:         ${policyConfig.policyParams?.maxVelocity1hPercent}% of balance`);
  console.log(`  Max per day:          ${policyConfig.policyParams?.maxVelocity24hPercent}% of balance`);
  console.log(`  Min vendor trust:     ${policyConfig.policyParams?.minVendorTrust}/100`);

  console.log("\n───────────────────────────────────────────────────────────────\n");

  // =========================================================================
  // STEP 3: Make Request (Full ZK Flow)
  // =========================================================================
  console.log("🚀 STEP 3: Making request with ZK proofs...\n");
  console.log("Request:");
  console.log(`  Prompt: "Hello, world!"`);
  console.log(`  Tier:   Pro (tier 1)\n`);

  try {
    const response = await agent.request(
      "POST",
      API_URL,
      {
        prompt: "Hello, world!",
        tier: 1,
      },
      policyConfig
    );

    console.log("\n───────────────────────────────────────────────────────────────\n");

    if (response.ok) {
      const result = await response.json();

      console.log("✅ SUCCESS! Transaction completed with full ZK trust stack\n");

      console.log("Response:");
      console.log(`  ${JSON.stringify(result, null, 2)}\n`);

      console.log("═══════════════════════════════════════════════════════════════");
      console.log("Trust Guarantees Achieved:");
      console.log("═══════════════════════════════════════════════════════════════");
      console.log("\n✅ Fair-Pricing Proof (Seller → Agent)");
      console.log("   The server cryptographically proved:");
      console.log("   - Price was computed correctly per public tariff");
      console.log("   - No price manipulation occurred");
      console.log("   - Agent verified this proof before paying\n");

      console.log("✅ Agent-Authorization Proof (Agent → Server)");
      console.log("   The agent cryptographically proved:");
      console.log("   - Authorized to spend this amount per policy");
      console.log("   - No policy violations");
      console.log("   - Server verified this proof before processing\n");

      console.log("✅ Privacy Preserved");
      console.log("   Agent's private data remained hidden:");
      console.log("   - Balance: $10.00 (NOT revealed to server)");
      console.log("   - Velocity: $0.20/1h, $0.50/24h (NOT revealed)");
      console.log("   - Trust score: 80/100 (NOT revealed)");
      console.log("   - Policy limits: 10%, 5%, 20% (NOT revealed)\n");

      console.log("═══════════════════════════════════════════════════════════════\n");
    } else {
      console.log(`❌ Request failed: ${response.status} ${response.statusText}\n`);
      const error = await response.json();
      console.log("Error details:");
      console.log(`  ${JSON.stringify(error, null, 2)}\n`);
    }
  } catch (error) {
    console.log("\n───────────────────────────────────────────────────────────────\n");
    console.log(`❌ ERROR: ${error}\n`);

    if (error instanceof Error && error.message.includes("policy violation")) {
      console.log("This error means the agent's zero-knowledge proof showed");
      console.log("that the transaction would violate the spending policy.");
      console.log("The agent prevented itself from overspending - this is good!\n");
    }
  }

  console.log("═══════════════════════════════════════════════════════════════");
  console.log("Demo Complete!");
  console.log("═══════════════════════════════════════════════════════════════\n");
}

// Run demo
runDemo().catch((error) => {
  console.error("Demo failed:", error);
  process.exit(1);
});
