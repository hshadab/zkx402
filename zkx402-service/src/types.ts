/**
 * ZKx402 Type Definitions
 *
 * Core types for the ZK-Fair-Pricing service
 */

export interface PublicTariff {
  /** Pricing tiers: 0=basic, 1=pro, 2=enterprise */
  tiers: {
    basic: TierPricing;
    pro: TierPricing;
    enterprise: TierPricing;
  };
  /** Surge/time-of-day multiplier in basis points (10000 = 1.0x) */
  multiplier: number;
}

export interface TierPricing {
  /** Base price in micro-units (e.g., $0.01 = 10000) */
  basePrice: bigint;
  /** Per-unit price in micro-units (e.g., $0.0001 = 100) */
  perUnitPrice: bigint;
}

export interface PricingRequest {
  /** Number of tokens/units consumed */
  tokens: bigint;
  /** Pricing tier (0=basic, 1=pro, 2=enterprise) */
  tier: 0 | 1 | 2;
  /** Optional metadata hash (for privacy) */
  metadataHash?: string;
}

export interface PricingProof {
  /** Final computed price in micro-units */
  price: bigint;
  /** Base64-encoded ZK proof (serialized SNARK) */
  proof: string;
  /** Proof type identifier */
  proofType: "zkengine-wasm";
  /** Public inputs used in proof */
  publicInputs: {
    tokens: bigint;
    tier: number;
    tariff: PublicTariff;
  };
}

export interface X402Challenge {
  /** HTTP 402 status */
  status: 402;
  /** Payment specification: chain:asset:amount */
  acceptPayment: string;
  /** ZK proof of fair pricing */
  pricingProof: string;
}

/**
 * Proof generation result from Rust prover
 */
export interface ProofGenerationResult {
  success: boolean;
  proof?: {
    snark: string; // Serialized proof
    instance: string; // Public instance
    price: bigint;
  };
  error?: string;
  stats?: {
    provingTimeMs: number;
    verifyingTimeMs: number;
    stepSize: number;
  };
}
