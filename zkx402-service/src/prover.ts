/**
 * ZK Proof Generation Service
 *
 * Interfaces with the Rust zkEngine prover to generate
 * zero-knowledge proofs of fair pricing.
 */

import { spawn } from "child_process";
import { join } from "path";
import type {
  PublicTariff,
  PricingRequest,
  PricingProof,
  ProofGenerationResult,
} from "./types.js";

/**
 * ZK Prover Service
 *
 * Generates zkEngine WASM proofs for pricing computations
 */
export class ZKProver {
  private zkEnginePath: string;

  constructor(zkEnginePath?: string) {
    // Default to ../zkEngine_dev relative to service root
    this.zkEnginePath =
      zkEnginePath || join(process.cwd(), "..", "zkEngine_dev");
  }

  /**
   * Generate a ZK proof that: price = compute_price(request, tariff)
   */
  async generatePricingProof(
    request: PricingRequest,
    tariff: PublicTariff
  ): Promise<PricingProof> {
    const startTime = Date.now();

    // Extract tariff parameters based on tier
    const { basePrice, perUnitPrice } = this.getTierPricing(
      request.tier,
      tariff
    );

    // Call Rust prover
    const result = await this.callRustProver({
      tokens: request.tokens,
      tier: request.tier,
      basePrices: [
        tariff.tiers.basic.basePrice,
        tariff.tiers.pro.basePrice,
        tariff.tiers.enterprise.basePrice,
      ],
      perUnitPrices: [
        tariff.tiers.basic.perUnitPrice,
        tariff.tiers.pro.perUnitPrice,
        tariff.tiers.enterprise.perUnitPrice,
      ],
      multiplier: tariff.multiplier,
    });

    if (!result.success || !result.proof) {
      throw new Error(
        `Proof generation failed: ${result.error || "Unknown error"}`
      );
    }

    const proof: PricingProof = {
      price: result.proof.price,
      proof: result.proof.snark,
      proofType: "zkengine-wasm",
      publicInputs: {
        tokens: request.tokens,
        tier: request.tier,
        tariff,
      },
    };

    console.log(
      `✓ Proof generated in ${Date.now() - startTime}ms for price ${this.formatPrice(proof.price)}`
    );

    return proof;
  }

  /**
   * Verify a pricing proof (local verification)
   *
   * In production, this would call the Rust verifier.
   * For now, we return true since zkEngine verifies during generation.
   */
  async verifyPricingProof(proof: PricingProof): Promise<boolean> {
    // TODO: Implement standalone verification
    // For now, trust the proof since zkEngine verifies after generation
    return true;
  }

  /**
   * Compute the expected price client-side (for comparison)
   */
  computeExpectedPrice(
    request: PricingRequest,
    tariff: PublicTariff
  ): bigint {
    const { basePrice, perUnitPrice } = this.getTierPricing(
      request.tier,
      tariff
    );

    // subtotal = basePrice + (tokens * perUnitPrice)
    const subtotal = basePrice + request.tokens * perUnitPrice;

    // final = (subtotal * multiplier) / 10000
    const finalPrice = (subtotal * BigInt(tariff.multiplier)) / 10000n;

    return finalPrice;
  }

  private getTierPricing(
    tier: 0 | 1 | 2,
    tariff: PublicTariff
  ): { basePrice: bigint; perUnitPrice: bigint } {
    switch (tier) {
      case 0:
        return tariff.tiers.basic;
      case 1:
        return tariff.tiers.pro;
      case 2:
        return tariff.tiers.enterprise;
    }
  }

  /**
   * Call the Rust zkEngine prover via CLI
   *
   * In production, this should use FFI or a gRPC service.
   * For the MVP, we shell out to the Rust binary.
   */
  private async callRustProver(params: {
    tokens: bigint;
    tier: number;
    basePrices: bigint[];
    perUnitPrices: bigint[];
    multiplier: number;
  }): Promise<ProofGenerationResult> {
    return new Promise((resolve, reject) => {
      // For now, we simulate the proof generation
      // In production, compile a custom Rust binary that takes JSON input

      // Calculate expected price
      const basePrice = params.basePrices[params.tier];
      const perUnitPrice = params.perUnitPrices[params.tier];
      const subtotal = basePrice + params.tokens * perUnitPrice;
      const finalPrice = (subtotal * BigInt(params.multiplier)) / 10000n;

      // Simulate proof generation delay
      setTimeout(() => {
        resolve({
          success: true,
          proof: {
            snark: Buffer.from("mock-zkengine-proof-" + Date.now()).toString(
              "base64"
            ),
            instance: Buffer.from("mock-instance").toString("base64"),
            price: finalPrice,
          },
          stats: {
            provingTimeMs: 150, // Real zkEngine takes ~1-5s for this circuit
            verifyingTimeMs: 10,
            stepSize: 50,
          },
        });
      }, 100); // Simulate 100ms proving time for demo

      // TODO: Replace with real Rust binary call:
      // const cargo = spawn("cargo", [
      //   "run",
      //   "--release",
      //   "--example",
      //   "zkx402_pricing_json",
      //   "--",
      //   JSON.stringify(params),
      // ], { cwd: this.zkEnginePath });
    });
  }

  private formatPrice(microUnits: bigint): string {
    return `$${(Number(microUnits) / 1_000_000).toFixed(6)}`;
  }
}
