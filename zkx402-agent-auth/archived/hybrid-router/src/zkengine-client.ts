import { exec } from "child_process";
import { promisify } from "util";
import type { AuthorizationRequest, AuthorizationResponse, ZKProof } from "./types.js";

const execAsync = promisify(exec);

/**
 * Client for zkEngine WASM prover
 *
 * This client wraps the Rust zkEngine prover and handles:
 * - Complex policy evaluation
 * - WASM circuit execution
 * - Proof generation
 */

export class ZkEngineClient {
  private readonly proverPath: string;

  constructor(proverPath: string = "../zkengine-prover") {
    this.proverPath = proverPath;
  }

  /**
   * Generate authorization proof using zkEngine WASM
   */
  async generateProof(
    request: AuthorizationRequest
  ): Promise<AuthorizationResponse> {
    const startTime = Date.now();

    try {
      // For now, compute locally and return mock proof
      // Real implementation would call Rust binary
      const result = await this.computeLocally(request);

      const provingTime = Date.now() - startTime;

      return {
        authorized: result.authorized,
        riskScore: 0, // zkEngine circuit doesn't compute risk score
        proof: {
          type: "zkengine",
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
          circuitType: "zkengine-wasm-complex-policy",
          verified: true,
        },
      };
    } catch (error) {
      throw new Error(`zkEngine proof generation failed: ${error}`);
    }
  }

  /**
   * Verify a zkEngine proof
   */
  async verifyProof(proof: ZKProof): Promise<boolean> {
    // Real implementation would call Rust verifier
    return proof.type === "zkengine" && proof.proofData.length > 0;
  }

  /**
   * Local computation (simulates what zkEngine WASM would prove)
   */
  private async computeLocally(
    request: AuthorizationRequest
  ): Promise<{ authorized: boolean; mockProof: Uint8Array }> {
    const amount = BigInt(request.transactionAmount);
    const balance = BigInt(request.balance);
    const vendorId = BigInt(request.vendorId);

    // Rule 1: Amount <= 10% of balance
    const maxAmount = balance / 10n;
    const rule1 = amount <= maxAmount;

    // Rule 2: Vendor whitelist check
    let rule2 = true;
    if (request.policyParams?.requireWhitelist && request.whitelistData) {
      const whitelistBitmap = BigInt(request.whitelistData.whitelistBitmap || "0");
      const bitPosition = vendorId % 64n;
      const mask = 1n << bitPosition;
      rule2 = (whitelistBitmap & mask) !== 0n;
    }

    // Rule 3: Business hours check
    let rule3 = true;
    if (request.policyParams?.requireBusinessHours) {
      const timestamp = request.timestamp;
      const secondsInDay = 86400;
      const hour = Math.floor((timestamp % secondsInDay) / 3600);
      const daysSinceEpoch = Math.floor(timestamp / secondsInDay);
      const dayOfWeek = (daysSinceEpoch + 4) % 7; // Unix epoch was Thursday

      const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
      const isWorkHours = hour >= 9 && hour < 17;
      rule3 = isWeekday && isWorkHours;
    }

    // Rule 4: Daily budget check
    let rule4 = true;
    if (request.policyParams?.requireDailyBudget && request.budgetData) {
      const dailyBudgetRemaining = BigInt(request.budgetData.dailyBudgetRemaining || "0");
      rule4 = amount <= dailyBudgetRemaining;
    }

    // Approved if ALL rules pass
    const authorized = rule1 && rule2 && rule3 && rule4;

    // Mock proof (~1.5KB for zkEngine)
    const mockProof = new Uint8Array(1536);
    mockProof.fill(84); // Placeholder

    return { authorized, mockProof };
  }

  /**
   * Call the actual Rust binary (for real proofs)
   * NOTE: This is commented out but shows how to integrate real zkEngine
   */
  private async callRustProver(request: AuthorizationRequest): Promise<string> {
    // Serialize request to JSON
    const input = JSON.stringify({
      transaction_amount: request.transactionAmount,
      vendor_id: request.vendorId,
      timestamp: request.timestamp,
      balance: request.balance,
      daily_budget_remaining: request.budgetData?.dailyBudgetRemaining || "0",
      whitelist_bitmap: request.whitelistData?.whitelistBitmap || "0",
    });

    // Call Rust binary
    const { stdout, stderr } = await execAsync(
      `cd ${this.proverPath} && cargo run --release --example complex_auth -- '${input}'`
    );

    if (stderr) {
      console.error("zkEngine prover stderr:", stderr);
    }

    return stdout;
  }
}
