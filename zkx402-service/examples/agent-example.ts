/**
 * Agent Integration Example
 *
 * Demonstrates how AI agents discover and consume ZK-verified x402 APIs
 */

import { ZKx402Agent } from "../src/agent-sdk.js";

// ==============================================================================
// Example 1: Discovery
// ==============================================================================

async function example1Discovery() {
  console.log("\n=== Example 1: Discover ZK-verified x402 API ===\n");

  const agent = new ZKx402Agent({ verbose: true });

  // Discover capabilities
  const discovery = await agent.discover("http://localhost:3402/api/llm/generate");

  console.log("\n✅ Discovery Result:");
  console.log("   Supports x402:", discovery.supportsX402);
  console.log("   Supports ZK Pricing:", discovery.supportsZKPricing);
  console.log("   Schemes:", discovery.schemes);
  console.log("   Networks:", discovery.networks);
  console.log("   Tariff Hash:", discovery.tariffHash);
  console.log("   Proof Type:", discovery.proofType);
}

// ==============================================================================
// Example 2: Automatic Payment Flow
// ==============================================================================

async function example2AutoPayment() {
  console.log("\n=== Example 2: Automatic Payment Flow ===\n");

  const agent = new ZKx402Agent({
    verbose: true,
    autoRetry: true,
    // Mock wallet (in production, use real wallet)
    wallet: {
      pay: async (amount, to, network) => {
        console.log(`\n💳 [Wallet] Paying ${amount} micro-USDC to ${to} on ${network}`);
        // In production, sign EIP-712 authorization and return signature
        return `mock-signature-${Date.now()}`;
      },
    },
  });

  try {
    // Make request (agent handles 402 automatically)
    const response = await agent.request("POST", "http://localhost:3402/api/llm/generate", {
      prompt: "What is zero-knowledge proof?",
      tier: 1,
    });

    console.log("\n✅ Response received:");
    console.log("   Result:", response.result);
    console.log("   ZK Verified:", response.x402.zkVerified);
    console.log("   Transaction:", response.x402.transactionHash);
  } catch (error: any) {
    console.error("\n❌ Error:", error.message);
  }
}

// ==============================================================================
// Example 3: Manual Payment Flow (Step-by-Step)
// ==============================================================================

async function example3ManualPayment() {
  console.log("\n=== Example 3: Manual Payment Flow ===\n");

  // Step 1: Initial request (no payment)
  console.log("📤 [Step 1] Making initial request...");
  const response1 = await fetch("http://localhost:3402/api/llm/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: "Explain x402 protocol",
      tier: 1,
    }),
  });

  console.log(`   Status: ${response1.status} ${response1.statusText}`);

  if (response1.status !== 402) {
    console.log("   ⚠️ Expected 402, got different status");
    return;
  }

  // Step 2: Parse 402 response
  console.log("\n💰 [Step 2] Parsing payment requirements...");
  const payment402 = await response1.json();
  console.log("   x402 Version:", payment402.x402Version);
  console.log("   Accepts:", payment402.accepts.map((a: any) => a.scheme));

  const zkRequirement = payment402.accepts.find((a: any) => a.scheme === "zkproof");
  console.log("\n   ZK Requirement:");
  console.log("     Network:", zkRequirement.network);
  console.log("     Amount:", zkRequirement.maxAmountRequired, "micro-USDC");
  console.log("     Recipient:", zkRequirement.payTo);
  console.log("     Tariff Hash:", zkRequirement.extra.tariffHash);

  // Step 3: Verify ZK pricing proof
  console.log("\n🔐 [Step 3] Verifying ZK pricing proof...");
  const zkProofHeader = response1.headers.get("X-Pricing-Proof");
  if (zkProofHeader) {
    const zkProof = JSON.parse(zkProofHeader);
    console.log("   Proof Type:", zkProof.type);
    console.log("   Price:", zkProof.price, "micro-USDC");
    console.log("   Inputs:", zkProof.inputs);

    // Verify price matches tariff
    const tariff = zkRequirement.extra.tariff;
    const tier = zkProof.inputs.tier;
    const tokens = BigInt(zkProof.inputs.tokens);

    const tierPricing = tier === 0 ? tariff.tiers.basic : tier === 1 ? tariff.tiers.pro : tariff.tiers.enterprise;
    const basePrice = BigInt(tierPricing.basePrice);
    const perUnitPrice = BigInt(tierPricing.perUnitPrice);

    const expectedPrice = basePrice + tokens * perUnitPrice;
    const actualPrice = BigInt(zkProof.price);

    if (expectedPrice === actualPrice) {
      console.log("   ✅ Proof verified: Price matches public tariff");
    } else {
      console.log(`   ❌ Proof invalid: Expected ${expectedPrice}, got ${actualPrice}`);
      return;
    }
  }

  // Step 4: Generate payment
  console.log("\n💳 [Step 4] Generating payment...");
  const paymentPayload = {
    x402Version: 1,
    scheme: "zkproof",
    network: zkRequirement.network,
    payload: {
      zkProof: zkProofHeader ? JSON.parse(zkProofHeader).proof : undefined,
      verificationToken: `mock-payment-${Date.now()}`,
    },
  };

  const paymentHeader = Buffer.from(JSON.stringify(paymentPayload)).toString("base64");
  console.log("   Payment header generated");

  // Step 5: Retry with payment
  console.log("\n🔄 [Step 5] Retrying with X-Payment header...");
  const response2 = await fetch("http://localhost:3402/api/llm/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Payment": paymentHeader,
    },
    body: JSON.stringify({
      prompt: "Explain x402 protocol",
      tier: 1,
    }),
  });

  console.log(`   Status: ${response2.status} ${response2.statusText}`);

  if (response2.ok) {
    const result = await response2.json();
    console.log("\n✅ [Step 6] Payment successful!");
    console.log("   Result:", result.result);
    console.log("   ZK Verified:", result.x402.zkVerified);

    const settlementHeader = response2.headers.get("X-Payment-Response");
    if (settlementHeader) {
      const settlement = JSON.parse(Buffer.from(settlementHeader, "base64").toString("utf-8"));
      console.log("\n📋 Settlement Details:");
      console.log("   Transaction:", settlement.transactionHash);
      console.log("   Status:", settlement.status);
      console.log("   Settled At:", settlement.settledAt);
    }
  } else {
    console.log("   ❌ Payment failed");
  }
}

// ==============================================================================
// Example 4: Comparing Prices Across Tiers
// ==============================================================================

async function example4ComparePrices() {
  console.log("\n=== Example 4: Compare Prices Across Tiers ===\n");

  const prompt = "Hello world!";
  const estimatedTokens = Math.ceil(prompt.length / 4);

  console.log(`Prompt: "${prompt}"`);
  console.log(`Estimated tokens: ${estimatedTokens}\n`);

  for (const tier of [0, 1, 2]) {
    const response = await fetch("http://localhost:3402/api/llm/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, tier }),
    });

    if (response.status === 402) {
      const zkProofHeader = response.headers.get("X-Pricing-Proof");
      if (zkProofHeader) {
        const zkProof = JSON.parse(zkProofHeader);
        const price = parseInt(zkProof.price);
        const tierName = tier === 0 ? "Basic" : tier === 1 ? "Pro" : "Enterprise";

        console.log(`${tierName} tier: $${(price / 1_000_000).toFixed(6)} (${price} micro-USDC)`);
      }
    }
  }

  console.log("\n✅ Notice: Higher tiers cost more due to different base + per-token pricing");
}

// ==============================================================================
// Run Examples
// ==============================================================================

async function main() {
  console.log("╔═══════════════════════════════════════════════════════════════╗");
  console.log("║                                                               ║");
  console.log("║   ZKx402 Agent Integration Examples                          ║");
  console.log("║                                                               ║");
  console.log("╚═══════════════════════════════════════════════════════════════╝");

  try {
    await example1Discovery();
    await example2AutoPayment();
    await example3ManualPayment();
    await example4ComparePrices();

    console.log("\n" + "=".repeat(65));
    console.log("✅ All examples completed successfully!");
    console.log("=".repeat(65) + "\n");
  } catch (error: any) {
    console.error("\n❌ Error running examples:", error.message);
    console.error(error.stack);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main };
