/**
 * Common types for the hybrid authorization router
 */

export type PolicyType = "simple" | "complex";
export type ProverBackend = "jolt" | "zkengine";

/**
 * Authorization request (from agent)
 */
export interface AuthorizationRequest {
  // Public inputs
  transactionAmount: string; // micro-USDC
  vendorId: string; // hash or identifier
  timestamp: number; // Unix timestamp

  // Private inputs (hidden by ZK proof)
  balance: string; // micro-USDC
  velocityData?: VelocityData;
  budgetData?: BudgetData;
  whitelistData?: WhitelistData;

  // Policy configuration
  policyType: PolicyType;
  policyParams?: PolicyParams;
}

/**
 * Velocity tracking data (for simple policies)
 */
export interface VelocityData {
  velocity1h: string; // spending in last hour
  velocity24h: string; // spending in last 24 hours
  vendorTrustScore: number; // 0-100
}

/**
 * Budget tracking data (for complex policies)
 */
export interface BudgetData {
  dailyBudgetRemaining: string; // micro-USDC remaining today
  monthlyBudgetRemaining?: string;
}

/**
 * Whitelist data (for complex policies)
 */
export interface WhitelistData {
  whitelistBitmap: string; // 64-bit bitmap as hex string
}

/**
 * Policy parameters
 */
export interface PolicyParams {
  // Simple policy params
  maxSingleTxPercent?: number; // e.g., 10 = 10% of balance
  maxVelocity1hPercent?: number;
  maxVelocity24hPercent?: number;
  minVendorTrust?: number;

  // Complex policy params
  requireBusinessHours?: boolean;
  requireWhitelist?: boolean;
  requireDailyBudget?: boolean;
}

/**
 * Authorization response (with ZK proof)
 */
export interface AuthorizationResponse {
  authorized: boolean;
  riskScore: number; // 0-100
  proof: ZKProof;
  metadata: ProofMetadata;
}

/**
 * Zero-knowledge proof data
 */
export interface ZKProof {
  type: ProverBackend;
  proofData: string; // base64 encoded proof
  publicInputs: Record<string, any>;
}

/**
 * Proof generation metadata
 */
export interface ProofMetadata {
  provingTimeMs: number;
  proofSizeBytes: number;
  circuitType: string;
  verified: boolean;
}

/**
 * Policy classification result
 */
export interface PolicyClassification {
  policyType: PolicyType;
  backend: ProverBackend;
  reason: string;
}
