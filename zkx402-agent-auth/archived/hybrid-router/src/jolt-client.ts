import { exec } from "child_process";
import { promisify } from "util";
import type { AuthorizationRequest, AuthorizationResponse, ZKProof } from "./types.js";

const execAsync = promisify(exec);

/**
 * Client for JOLT zkVM prover
 *
 * This client wraps the Rust JOLT prover binary and handles:
 * - Input serialization
 * - Proof generation via CLI
 * - Output parsing
 */

export class JoltClient {
  private readonly proverPath: string;

  constructor(proverPath: string = "../jolt-prover") {
    this.proverPath = proverPath;
  }

  /**
   * Generate authorization proof using JOLT zkVM
   */
  async generateProof(
    request: AuthorizationRequest
  ): Promise<AuthorizationResponse> {
    const startTime = Date.now();

    try {
      // For now, we'll compute locally and return a mock proof
      // Real implementation would call the Rust binary with cargo run
      const result = await this.computeLocally(request);

      const provingTime = Date.now() - startTime;

      return {
        authorized: result.authorized,
        riskScore: result.riskScore,
        proof: {
          type: "jolt",
          proofData: Buffer.from(result.mockProof).toString("base64"),
          publicInputs: {
            transactionAmount: request.transactionAmount,
            vendorId: request.vendorId,
            timestamp: request.timestamp,
          },
        },
        metadata: {
          provingTimeMs: provingTime,
          proofSizeBytes: result.mockProof.length,
          circuitType: "jolt-zkvm-velocity-check",
          verified: true,
        },
      };
    } catch (error) {
      throw new Error(`JOLT proof generation failed: ${error}`);
    }
  }

  /**
   * Verify a JOLT proof
   */
  async verifyProof(proof: ZKProof): Promise<boolean> {
    // Real implementation would call Rust verifier
    // For now, just check proof structure
    return proof.type === "jolt" && proof.proofData.length > 0;
  }

  /**
   * Local computation (simulates what JOLT zkVM would prove)
   */
  private async computeLocally(
    request: AuthorizationRequest
  ): Promise<{ authorized: boolean; riskScore: number; mockProof: Uint8Array }> {
    const amount = BigInt(request.transactionAmount);
    const balance = BigInt(request.balance);

    const velocityData = request.velocityData || {
      velocity1h: "0",
      velocity24h: "0",
      vendorTrustScore: 50,
    };

    const policyParams = request.policyParams || {
      maxSingleTxPercent: 10,
      maxVelocity1hPercent: 5,
      maxVelocity24hPercent: 20,
      minVendorTrust: 50,
    };

    // Rule 1: Amount <= max% of balance
    const maxAmount = (balance * BigInt(policyParams.maxSingleTxPercent || 10)) / 100n;
    const rule1 = amount <= maxAmount;

    // Rule 2: 1h velocity <= max%
    const velocity1h = BigInt(velocityData.velocity1h);
    const maxVel1h = (balance * BigInt(policyParams.maxVelocity1hPercent || 5)) / 100n;
    const rule2 = velocity1h <= maxVel1h;

    // Rule 3: 24h velocity <= max%
    const velocity24h = BigInt(velocityData.velocity24h);
    const maxVel24h = (balance * BigInt(policyParams.maxVelocity24hPercent || 20)) / 100n;
    const rule3 = velocity24h <= maxVel24h;

    // Rule 4: Vendor trust >= minimum
    const rule4 = velocityData.vendorTrustScore >= (policyParams.minVendorTrust || 50);

    // Approved if ALL rules pass
    const authorized = rule1 && rule2 && rule3 && rule4;

    // Calculate risk score
    let risk = 0;
    if (!rule1) risk += 40;
    if (!rule2) risk += 30;
    if (!rule3) risk += 20;
    if (!rule4) risk += 10;

    // Mock proof (524 bytes for JOLT)
    const mockProof = new Uint8Array(524);
    mockProof.fill(42); // Placeholder proof data

    return { authorized, riskScore: risk, mockProof };
  }

  /**
   * Call the actual Rust binary with JOLT Atlas (for real proofs)
   */
  private async callRustProver(request: AuthorizationRequest): Promise<string> {
    // Serialize request to JSON
    const input = JSON.stringify({
      transaction_amount: request.transactionAmount,
      vendor_id: request.vendorId,
      timestamp: request.timestamp,
      balance: request.balance,
      velocity_1h: request.velocityData?.velocity1h || "0",
      velocity_24h: request.velocityData?.velocity24h || "0",
      vendor_trust_score: request.velocityData?.vendorTrustScore || 50,
      max_single_tx_percent: request.policyParams?.maxSingleTxPercent || 10,
      max_velocity_1h_percent: request.policyParams?.maxVelocity1hPercent || 5,
      max_velocity_24h_percent: request.policyParams?.maxVelocity24hPercent || 20,
      min_vendor_trust: request.policyParams?.minVendorTrust || 50,
    });

    // Call Rust binary
    const { stdout, stderr } = await execAsync(
      `cd ${this.proverPath} && cargo run --release --bin host -- '${input}'`
    );

    if (stderr) {
      console.error("JOLT prover stderr:", stderr);
    }

    return stdout;
  }
}
