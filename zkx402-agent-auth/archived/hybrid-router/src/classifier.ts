import type {
  AuthorizationRequest,
  PolicyClassification,
  PolicyType,
  ProverBackend,
} from "./types.js";

/**
 * Policy Complexity Classifier
 *
 * Determines whether to use JOLT zkVM (simple) or zkEngine WASM (complex)
 * based on the policy requirements.
 *
 * Decision criteria:
 *
 * Simple (JOLT) - Use when policy only needs:
 *   - Numeric comparisons (amount, velocity, thresholds)
 *   - Basic arithmetic (percentages, sums)
 *   - No string operations
 *   - No complex control flow
 *
 * Complex (zkEngine WASM) - Use when policy needs:
 *   - Whitelist/blacklist checking (bitmap operations)
 *   - Time-based rules (business hours, weekends)
 *   - Multi-condition boolean logic
 *   - String operations
 *   - IF/ELSE branching
 *
 * Performance trade-off:
 *   - JOLT: ~0.7s proving, 524 bytes, limited to simple policies
 *   - zkEngine: ~5-10s proving, ~1-2KB, supports all policies
 */

export class PolicyClassifier {
  /**
   * Classify a policy and determine which backend to use
   */
  static classify(request: AuthorizationRequest): PolicyClassification {
    // Explicit policy type override
    if (request.policyType === "complex") {
      return {
        policyType: "complex",
        backend: "zkengine",
        reason: "Explicit complex policy type specified",
      };
    }

    if (request.policyType === "simple") {
      return {
        policyType: "simple",
        backend: "jolt",
        reason: "Explicit simple policy type specified",
      };
    }

    // Auto-detect based on features

    // Check for complex features
    const hasWhitelist = !!request.whitelistData;
    const requiresBusinessHours = request.policyParams?.requireBusinessHours === true;
    const hasBudgetTracking = !!request.budgetData?.dailyBudgetRemaining;

    // If any complex feature is present, use zkEngine
    if (hasWhitelist || requiresBusinessHours || hasBudgetTracking) {
      const reasons: string[] = [];
      if (hasWhitelist) reasons.push("whitelist checking");
      if (requiresBusinessHours) reasons.push("business hours validation");
      if (hasBudgetTracking) reasons.push("daily budget tracking");

      return {
        policyType: "complex",
        backend: "zkengine",
        reason: `Complex features required: ${reasons.join(", ")}`,
      };
    }

    // Default to simple (JOLT) for basic velocity checks
    return {
      policyType: "simple",
      backend: "jolt",
      reason: "Simple numeric policy (velocity + trust checks only)",
    };
  }

  /**
   * Estimate proving time based on classification
   */
  static estimateProvingTime(classification: PolicyClassification): number {
    return classification.backend === "jolt" ? 700 : 6000; // ms
  }

  /**
   * Estimate proof size based on classification
   */
  static estimateProofSize(classification: PolicyClassification): number {
    return classification.backend === "jolt" ? 524 : 1536; // bytes
  }

  /**
   * Get human-readable description of the policy
   */
  static describe(classification: PolicyClassification): string {
    const backend = classification.backend === "jolt" ? "JOLT zkVM" : "zkEngine WASM";
    const time = this.estimateProvingTime(classification);
    const size = this.estimateProofSize(classification);

    return `${classification.policyType.toUpperCase()} policy using ${backend} (est. ${time}ms, ~${size} bytes)`;
  }
}
