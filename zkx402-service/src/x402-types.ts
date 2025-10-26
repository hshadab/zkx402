/**
 * x402 Protocol Type Definitions
 *
 * Based on: https://github.com/coinbase/x402
 * Spec-compliant types for x402 payment protocol
 */

/**
 * x402 Protocol Version
 */
export const X402_VERSION = 1;

/**
 * Payment scheme identifier
 */
export type PaymentScheme = "exact" | "zkproof" | string;

/**
 * Blockchain network identifier
 */
export type Network =
  | "base-sepolia"
  | "base"
  | "ethereum"
  | "polygon"
  | "avalanche"
  | string;

/**
 * Payment requirement in 402 response
 *
 * Spec: github.com/coinbase/x402
 */
export interface PaymentRequirement {
  /** Payment scheme (e.g., "exact", "zkproof") */
  scheme: PaymentScheme;

  /** Blockchain network */
  network: Network;

  /** Maximum amount required (in smallest unit, e.g., micro-USDC) */
  maxAmountRequired: string;

  /** Resource URL being purchased */
  resource: string;

  /** Human-readable description */
  description: string;

  /** Response MIME type */
  mimeType: string;

  /** Recipient wallet address */
  payTo: string;

  /** Payment timeout in seconds */
  maxTimeoutSeconds: number;

  /** Asset contract address (e.g., USDC on Base) */
  asset: string;

  /** Scheme-specific metadata */
  extra?: {
    /** ZK-specific: tariff commitment */
    tariffHash?: string;

    /** ZK-specific: proof verification key */
    verificationKey?: string;

    /** EIP-3009: token name */
    tokenName?: string;

    /** EIP-3009: token version */
    tokenVersion?: string;

    [key: string]: any;
  };
}

/**
 * 402 Payment Required response body
 */
export interface PaymentRequiredResponse {
  /** Protocol version */
  x402Version: number;

  /** Accepted payment methods */
  accepts: PaymentRequirement[];

  /** Error message (optional) */
  error?: string;
}

/**
 * X-PAYMENT header payload (base64-encoded JSON)
 */
export interface XPaymentHeader {
  /** Protocol version */
  x402Version: number;

  /** Selected payment scheme */
  scheme: PaymentScheme;

  /** Selected network */
  network: Network;

  /** Scheme-dependent payment payload */
  payload: {
    /** For "exact" scheme: EIP-712 signed authorization */
    authorization?: string;

    /** For "zkproof" scheme: ZK proof of payment */
    zkProof?: string;

    /** Facilitator verification token */
    verificationToken?: string;

    [key: string]: any;
  };
}

/**
 * X-PAYMENT-RESPONSE header payload (base64-encoded JSON)
 */
export interface XPaymentResponseHeader {
  /** Protocol version */
  x402Version: number;

  /** Transaction hash (if settled on-chain) */
  transactionHash?: string;

  /** Settlement status */
  status: "pending" | "confirmed" | "failed";

  /** Blockchain network */
  network: Network;

  /** Settlement timestamp */
  settledAt?: string;

  /** Additional metadata */
  metadata?: {
    /** ZK-specific: pricing proof */
    pricingProof?: string;

    /** ZK-specific: price verified flag */
    priceVerified?: boolean;

    [key: string]: any;
  };
}

/**
 * ZK-Proof Extension for x402
 *
 * Custom scheme: "zkproof"
 * Adds zero-knowledge pricing proofs to x402 protocol
 */
export interface ZKProofPaymentRequirement extends PaymentRequirement {
  scheme: "zkproof";

  extra: {
    /** Hash of public tariff (for verification) */
    tariffHash: string;

    /** zkEngine verification key (public parameters) */
    verificationKey: string;

    /** Proof type identifier */
    proofType: "zkengine-wasm" | "jolt-atlas";

    /** Public tariff (for agent verification) */
    tariff: {
      tiers: {
        basic: { basePrice: string; perUnitPrice: string };
        pro: { basePrice: string; perUnitPrice: string };
        enterprise: { basePrice: string; perUnitPrice: string };
      };
      multiplier: number;
    };

    /** Request metadata schema (for proof inputs) */
    metadataSchema: {
      tokens: "bigint";
      tier: "0 | 1 | 2";
    };
  };
}

/**
 * Facilitator API types
 */
export interface FacilitatorVerifyRequest {
  payment: XPaymentHeader;
  requirement: PaymentRequirement;
}

export interface FacilitatorVerifyResponse {
  valid: boolean;
  error?: string;
  metadata?: Record<string, any>;
}

export interface FacilitatorSettleRequest {
  payment: XPaymentHeader;
  requirement: PaymentRequirement;
  verificationToken: string;
}

export interface FacilitatorSettleResponse {
  transactionHash: string;
  status: "pending" | "confirmed";
  network: Network;
}

/**
 * Helper: Encode/decode X-PAYMENT header
 */
export function encodeXPaymentHeader(payload: XPaymentHeader): string {
  return Buffer.from(JSON.stringify(payload)).toString("base64");
}

export function decodeXPaymentHeader(header: string): XPaymentHeader {
  return JSON.parse(Buffer.from(header, "base64").toString("utf-8"));
}

/**
 * Helper: Encode/decode X-PAYMENT-RESPONSE header
 */
export function encodeXPaymentResponseHeader(
  payload: XPaymentResponseHeader
): string {
  return Buffer.from(JSON.stringify(payload)).toString("base64");
}

export function decodeXPaymentResponseHeader(
  header: string
): XPaymentResponseHeader {
  return JSON.parse(Buffer.from(header, "base64").toString("utf-8"));
}
