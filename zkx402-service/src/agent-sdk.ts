/**
 * ZKx402 Agent SDK
 *
 * Client library for AI agents to discover and consume ZK-verified x402 APIs
 *
 * Features:
 * - Automatic discovery via OPTIONS pre-flight
 * - Client-side ZK proof verification
 * - Payment flow handling
 * - Retry logic with X-PAYMENT header
 */

import type {
  PaymentRequiredResponse,
  PaymentRequirement,
  ZKProofPaymentRequirement,
  XPaymentHeader,
  XPaymentResponseHeader,
} from "./x402-types.js";
import {
  X402_VERSION,
  encodeXPaymentHeader,
  decodeXPaymentResponseHeader,
} from "./x402-types.js";
import type { PublicTariff } from "./types.js";
import { ZKProver } from "./prover.js";

export interface AgentSDKConfig {
  /** Agent wallet/payment provider */
  wallet?: {
    /** Sign and pay function */
    pay: (amount: bigint, to: string, network: string) => Promise<string>;
  };

  /** ZK proof verifier (optional, for client-side verification) */
  verifier?: ZKProver;

  /** Enable automatic retries on 402 */
  autoRetry?: boolean;

  /** Log discovery and verification details */
  verbose?: boolean;
}

export interface DiscoveryResult {
  /** Whether the API supports x402 payments */
  supportsX402: boolean;

  /** Whether the API supports ZK pricing proofs */
  supportsZKPricing: boolean;

  /** Supported payment schemes */
  schemes: string[];

  /** Supported networks */
  networks: string[];

  /** Tariff hash (if ZK-enabled) */
  tariffHash?: string;

  /** Proof type (if ZK-enabled) */
  proofType?: string;
}

export interface PaymentDetails {
  /** Amount in smallest unit (e.g., micro-USDC) */
  amount: bigint;

  /** Recipient address */
  recipient: string;

  /** Network */
  network: string;

  /** Asset contract address */
  asset: string;

  /** ZK pricing proof (if available) */
  zkProof?: {
    proof: string;
    verified: boolean;
    price: bigint;
  };
}

/**
 * ZKx402 Agent SDK
 *
 * Example usage:
 * ```typescript
 * const agent = new ZKx402Agent({ wallet: myWallet });
 *
 * // Discover API capabilities
 * const discovery = await agent.discover('https://api.example.com/llm');
 * console.log('Supports ZK pricing:', discovery.supportsZKPricing);
 *
 * // Make a paid request (handles 402 automatically)
 * const response = await agent.request('POST', 'https://api.example.com/llm', {
 *   prompt: 'Hello world',
 *   tier: 1,
 * });
 * ```
 */
export class ZKx402Agent {
  private config: AgentSDKConfig;
  private prover: ZKProver;

  constructor(config: AgentSDKConfig = {}) {
    this.config = config;
    this.prover = config.verifier || new ZKProver();
  }

  /**
   * Discover x402 + ZK capabilities via OPTIONS pre-flight
   */
  async discover(url: string): Promise<DiscoveryResult> {
    this.log(`[Discovery] Checking ${url}...`);

    try {
      const response = await fetch(url, {
        method: "OPTIONS",
        headers: {
          "Access-Control-Request-Method": "POST",
        },
      });

      const result: DiscoveryResult = {
        supportsX402: false,
        supportsZKPricing: false,
        schemes: [],
        networks: [],
      };

      // Check for x402 support
      const acceptsPayment = response.headers.get("X-Accepts-Payment");
      if (acceptsPayment) {
        result.supportsX402 = true;
        result.schemes = acceptsPayment.split(",").map((s) => s.trim());
      }

      const network = response.headers.get("X-Payment-Network");
      if (network) {
        result.networks = [network];
      }

      // Check for ZK pricing support
      const zkEnabled = response.headers.get("X-ZK-Pricing-Enabled");
      if (zkEnabled === "true") {
        result.supportsZKPricing = true;
        result.tariffHash = response.headers.get("X-Tariff-Hash") || undefined;
        result.proofType = response.headers.get("X-Proof-Type") || undefined;
      }

      this.log("[Discovery] Result:", result);
      return result;
    } catch (error) {
      this.log("[Discovery] Failed:", error);
      return {
        supportsX402: false,
        supportsZKPricing: false,
        schemes: [],
        networks: [],
      };
    }
  }

  /**
   * Make a request to a ZK-verified x402 API
   *
   * Handles the full flow:
   * 1. Initial request (may get 402)
   * 2. Verify ZK pricing proof
   * 3. Pay via wallet
   * 4. Retry with X-PAYMENT header
   */
  async request<T = any>(
    method: string,
    url: string,
    body?: any,
    headers?: Record<string, string>
  ): Promise<T> {
    this.log(`[Request] ${method} ${url}`);

    // Step 1: Initial request (no payment)
    const initialResponse = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    // If not 402, return directly
    if (initialResponse.status !== 402) {
      return await initialResponse.json();
    }

    // Step 2: Parse 402 response
    const payment402: PaymentRequiredResponse = await initialResponse.json();
    this.log("[402] Payment required:", payment402);

    if (!this.config.autoRetry && !this.config.wallet) {
      throw new Error(
        "Payment required but wallet not configured. Set autoRetry=true or provide wallet."
      );
    }

    // Step 3: Select payment requirement (prefer zkproof)
    const requirement = this.selectRequirement(payment402.accepts);
    this.log("[Payment] Selected requirement:", requirement);

    // Step 4: Verify ZK pricing proof (if available)
    const zkProofHeader = initialResponse.headers.get("X-Pricing-Proof");
    if (zkProofHeader && requirement.scheme === "zkproof") {
      const zkProof = JSON.parse(zkProofHeader);
      await this.verifyPricingProof(zkProof, requirement as ZKProofPaymentRequirement);
    }

    // Step 5: Generate payment
    const paymentHeader = await this.generatePayment(requirement, zkProofHeader);

    // Step 6: Retry with X-PAYMENT header
    this.log("[Retry] Retrying with payment...");
    const paidResponse = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-Payment": paymentHeader,
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!paidResponse.ok) {
      throw new Error(`Payment failed: ${paidResponse.statusText}`);
    }

    // Step 7: Parse settlement response
    const settlementHeader = paidResponse.headers.get("X-Payment-Response");
    if (settlementHeader) {
      const settlement = decodeXPaymentResponseHeader(settlementHeader);
      this.log("[Settlement]", settlement);
    }

    return await paidResponse.json();
  }

  /**
   * Select the best payment requirement from 402 response
   * Prefers zkproof > exact
   */
  private selectRequirement(requirements: PaymentRequirement[]): PaymentRequirement {
    // Prefer zkproof for ZK verification
    const zkReq = requirements.find((r) => r.scheme === "zkproof");
    if (zkReq) {
      this.log("[Select] Chose zkproof scheme");
      return zkReq;
    }

    // Fallback to exact
    const exactReq = requirements.find((r) => r.scheme === "exact");
    if (exactReq) {
      this.log("[Select] Chose exact scheme");
      return exactReq;
    }

    // Default to first
    return requirements[0];
  }

  /**
   * Verify ZK pricing proof client-side
   */
  private async verifyPricingProof(
    zkProofData: any,
    requirement: ZKProofPaymentRequirement
  ): Promise<void> {
    this.log("[ZK Verify] Checking pricing proof...");

    // Extract public tariff from requirement
    const tariff: PublicTariff = {
      tiers: {
        basic: {
          basePrice: BigInt(requirement.extra.tariff.tiers.basic.basePrice),
          perUnitPrice: BigInt(requirement.extra.tariff.tiers.basic.perUnitPrice),
        },
        pro: {
          basePrice: BigInt(requirement.extra.tariff.tiers.pro.basePrice),
          perUnitPrice: BigInt(requirement.extra.tariff.tiers.pro.perUnitPrice),
        },
        enterprise: {
          basePrice: BigInt(requirement.extra.tariff.tiers.enterprise.basePrice),
          perUnitPrice: BigInt(requirement.extra.tariff.tiers.enterprise.perUnitPrice),
        },
      },
      multiplier: requirement.extra.tariff.multiplier,
    };

    // Compute expected price from tariff
    const tokens = BigInt(zkProofData.inputs.tokens);
    const tier = zkProofData.inputs.tier as 0 | 1 | 2;

    const expectedPrice = this.prover.computeExpectedPrice(
      { tokens, tier },
      tariff
    );

    const actualPrice = BigInt(zkProofData.price);

    if (expectedPrice !== actualPrice) {
      throw new Error(
        `ZK proof verification failed: expected ${expectedPrice}, got ${actualPrice}`
      );
    }

    this.log("[ZK Verify] ✓ Pricing proof valid");
  }

  /**
   * Generate X-PAYMENT header
   */
  private async generatePayment(
    requirement: PaymentRequirement,
    zkProofHeader: string | null
  ): Promise<string> {
    const payload: XPaymentHeader = {
      x402Version: X402_VERSION,
      scheme: requirement.scheme,
      network: requirement.network,
      payload: {},
    };

    if (requirement.scheme === "zkproof" && zkProofHeader) {
      // Attach ZK proof
      const zkProof = JSON.parse(zkProofHeader);
      payload.payload.zkProof = zkProof.proof;
    }

    if (this.config.wallet) {
      // Sign payment via wallet
      const authorization = await this.config.wallet.pay(
        BigInt(requirement.maxAmountRequired),
        requirement.payTo,
        requirement.network
      );
      payload.payload.authorization = authorization;
    } else {
      // Mock payment for demo
      payload.payload.verificationToken = "mock-payment-" + Date.now();
    }

    return encodeXPaymentHeader(payload);
  }

  /**
   * Helper: Log if verbose mode enabled
   */
  private log(...args: any[]) {
    if (this.config.verbose) {
      console.log("[ZKx402Agent]", ...args);
    }
  }
}

/**
 * Convenience function for one-off requests
 */
export async function zkx402Request<T = any>(
  method: string,
  url: string,
  body?: any,
  config?: AgentSDKConfig
): Promise<T> {
  const agent = new ZKx402Agent({ ...config, autoRetry: true });
  return await agent.request<T>(method, url, body);
}
