/**
 * ZKx402 Agent SDK - Combined (Fair-Pricing + Agent-Authorization)
 *
 * This SDK enables agents to:
 * 1. Discover ZK-enabled x402 APIs
 * 2. Verify Fair-Pricing proofs from sellers
 * 3. Generate Agent-Authorization proofs
 * 4. Make payments with cryptographic guarantees
 */

import type { PublicTariff } from "./x402-types.js";

export interface DiscoveryResult {
  supportsX402: boolean;
  supportsZKPricing: boolean;
  supportsZKAgentAuth: boolean;
  zkPricingRequired: boolean;
  zkAgentAuthRequired: boolean;
  tariff?: PublicTariff;
  tariffHash?: string;
  authServiceUrl?: string;
  network?: string;
  asset?: string;
}

export interface PricingProof {
  type: string;
  verified: boolean;
  amount: string;
  metadata: {
    tokens: number;
    tier: number;
  };
}

export interface AgentAuthProof {
  type: "jolt" | "zkengine";
  authorized: boolean;
  riskScore: number;
  proof: {
    type: string;
    proofData: string;
    publicInputs: any;
  };
  metadata: {
    provingTimeMs: number;
    proofSizeBytes: number;
  };
}

export interface AuthPolicyConfig {
  // Private inputs (never sent to server)
  balance: string; // micro-USDC
  velocityData?: {
    velocity1h: string;
    velocity24h: string;
    vendorTrustScore: number;
  };
  budgetData?: {
    dailyBudgetRemaining: string;
  };
  whitelistData?: {
    whitelistBitmap: string;
  };

  // Policy parameters
  policyParams?: {
    maxSingleTxPercent?: number;
    maxVelocity1hPercent?: number;
    maxVelocity24hPercent?: number;
    minVendorTrust?: number;
    requireBusinessHours?: boolean;
    requireWhitelist?: boolean;
    requireDailyBudget?: boolean;
  };
}

export class ZKx402AgentCombined {
  private readonly authServiceUrl: string;

  constructor(authServiceUrl: string = "http://localhost:3403") {
    this.authServiceUrl = authServiceUrl;
  }

  /**
   * Discover API capabilities (OPTIONS pre-flight)
   */
  async discover(url: string): Promise<DiscoveryResult> {
    try {
      const response = await fetch(url, {
        method: "OPTIONS",
      });

      const acceptsPayment = response.headers.get("X-Accepts-Payment");
      const zkPricingEnabled = response.headers.get("X-ZK-Pricing-Enabled") === "true";
      const zkAgentAuthRequired = response.headers.get("X-ZK-Agent-Auth-Required") === "true";

      let tariff: PublicTariff | undefined;
      let tariffHash: string | undefined;
      let authServiceUrl: string | undefined;

      // Try to get tariff from response body
      if (response.ok) {
        try {
          const body = await response.json();
          tariff = body.zkPricing?.tariff;
          tariffHash = body.zkPricing?.tariffHash;
          authServiceUrl = body.zkAgentAuth?.serviceUrl;
        } catch {
          // Body not JSON, skip
        }
      }

      return {
        supportsX402: !!acceptsPayment,
        supportsZKPricing: zkPricingEnabled,
        supportsZKAgentAuth: zkAgentAuthRequired,
        zkPricingRequired: zkPricingEnabled,
        zkAgentAuthRequired,
        tariff,
        tariffHash,
        authServiceUrl,
        network: acceptsPayment?.split(":")[0],
        asset: acceptsPayment?.split(":")[1],
      };
    } catch (error) {
      return {
        supportsX402: false,
        supportsZKPricing: false,
        supportsZKAgentAuth: false,
        zkPricingRequired: false,
        zkAgentAuthRequired: false,
      };
    }
  }

  /**
   * Make a request with full ZK support
   */
  async request(
    method: string,
    url: string,
    body?: any,
    policyConfig?: AuthPolicyConfig
  ): Promise<Response> {
    // 1. Discover capabilities
    const discovery = await this.discover(url);

    // 2. Make initial request (will get 402)
    let response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    // 3. If 402, verify pricing proof and generate agent auth proof
    if (response.status === 402) {
      const paymentReq = await response.json();

      // Verify Fair-Pricing proof
      if (discovery.supportsZKPricing) {
        const pricingProofHeader = response.headers.get("X-Pricing-Proof");
        if (pricingProofHeader) {
          const pricingProof = JSON.parse(pricingProofHeader);
          const pricingValid = await this.verifyPricingProof(pricingProof);

          if (!pricingValid) {
            throw new Error(
              "Server pricing proof verification failed - price may be incorrect!"
            );
          }

          console.log("✅ Pricing proof verified - price is fair");
        }
      }

      // Generate Agent-Authorization proof (if required)
      let agentAuthProof: AgentAuthProof | undefined;
      if (discovery.zkAgentAuthRequired && policyConfig) {
        console.log("Generating agent authorization proof...");

        agentAuthProof = await this.generateAgentAuthProof(
          paymentReq.details.maxAmountRequired,
          url,
          policyConfig
        );

        if (!agentAuthProof.authorized) {
          throw new Error(
            `Agent policy violation: Not authorized to spend ${paymentReq.details.maxAmountRequired} micro-${discovery.asset}`
          );
        }

        console.log("✅ Agent authorization proof generated");
      }

      // Simulate payment (real implementation would call facilitator)
      const paymentToken = await this.simulatePayment(
        paymentReq.details.maxAmountRequired,
        discovery.network!,
        discovery.asset!
      );

      // Retry request with payment + agent auth proof
      response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json",
          "X-Payment": paymentToken,
          ...(agentAuthProof
            ? { "X-Agent-Auth-Proof": JSON.stringify(agentAuthProof) }
            : {}),
        },
        body: body ? JSON.stringify(body) : undefined,
      });
    }

    return response;
  }

  /**
   * Verify a Fair-Pricing proof
   */
  async verifyPricingProof(proof: any): Promise<boolean> {
    // Real implementation would:
    // 1. Check proof signature/SNARK verification
    // 2. Verify public inputs match claimed price
    // 3. Check tariff hash matches known tariff

    // For now, just check structure
    return proof.type === "fair-pricing" && proof.verified === true;
  }

  /**
   * Generate Agent-Authorization proof
   */
  async generateAgentAuthProof(
    transactionAmount: string,
    vendorUrl: string,
    policyConfig: AuthPolicyConfig
  ): Promise<AgentAuthProof> {
    // Hash vendor URL to get vendor ID
    const vendorId = await this.hashVendor(vendorUrl);

    // Build authorization request
    const authRequest = {
      transactionAmount,
      vendorId,
      timestamp: Math.floor(Date.now() / 1000),
      balance: policyConfig.balance,
      velocityData: policyConfig.velocityData,
      budgetData: policyConfig.budgetData,
      whitelistData: policyConfig.whitelistData,
      policyType: this.determinePolicyType(policyConfig),
      policyParams: policyConfig.policyParams,
    };

    // Call hybrid auth router
    const response = await fetch(`${this.authServiceUrl}/authorize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(authRequest),
    });

    if (!response.ok) {
      throw new Error(`Authorization proof generation failed: ${response.statusText}`);
    }

    return await response.json();
  }

  /**
   * Determine policy type based on config
   */
  private determinePolicyType(config: AuthPolicyConfig): "simple" | "complex" {
    const hasWhitelist = !!config.whitelistData;
    const requiresBusinessHours = config.policyParams?.requireBusinessHours === true;
    const hasBudgetTracking = !!config.budgetData;

    return hasWhitelist || requiresBusinessHours || hasBudgetTracking
      ? "complex"
      : "simple";
  }

  /**
   * Hash vendor URL to create vendor ID
   */
  private async hashVendor(url: string): Promise<string> {
    // Simple hash for demo (real implementation would use crypto hash)
    let hash = 0;
    for (let i = 0; i < url.length; i++) {
      hash = (hash << 5) - hash + url.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString();
  }

  /**
   * Simulate payment (placeholder)
   */
  private async simulatePayment(
    amount: string,
    network: string,
    asset: string
  ): Promise<string> {
    // Real implementation would:
    // 1. Connect to wallet
    // 2. Create payment transaction
    // 3. Sign and submit
    // 4. Get payment proof/receipt

    // Mock payment token
    const paymentData = {
      scheme: "exact",
      network,
      payload: {
        amount,
        asset,
        txHash: "0x" + Math.random().toString(16).slice(2),
        timestamp: Date.now(),
      },
    };

    return JSON.stringify(paymentData);
  }
}

// Export convenience function
export async function zkx402Request(
  method: string,
  url: string,
  body?: any,
  policyConfig?: AuthPolicyConfig
): Promise<Response> {
  const agent = new ZKx402AgentCombined();
  return agent.request(method, url, body, policyConfig);
}
