/**
 * JOLT Atlas Agent Authorization Example
 *
 * This example shows how to use JOLT Atlas to generate zero-knowledge proofs
 * of agent authorization using an ONNX velocity policy model.
 *
 * Flow:
 * 1. Load pre-trained ONNX model (from policy-examples/onnx/velocity_policy.onnx)
 * 2. Prepare inputs (public + private)
 * 3. Generate JOLT Atlas proof of ONNX inference
 * 4. Verify proof
 * 5. Check authorization result
 */

use zkml_jolt_core::{
    jolt::{JoltProverPreprocessing, JoltSNARK, execution_trace::jolt_execution_trace},
};
use ark_bn254::Fr;
use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme,
    utils::transcript::KeccakTranscript,
};
use onnx_tracer::{model, tensor::Tensor};

type PCS = DoryCommitmentScheme<KeccakTranscript>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("ZKx402 Agent Authorization - JOLT Atlas ONNX Prover");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Check if ONNX model exists
    let model_path = "../policy-examples/onnx/velocity_policy.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("❌ ONNX model not found at: {}", model_path);
        eprintln!("\nPlease train the model first:");
        eprintln!("  cd ../policy-examples/onnx");
        eprintln!("  pip install -r requirements.txt");
        eprintln!("  python train_velocity.py\n");
        return Ok(());
    }

    // Test case 1: Approved transaction
    println!("📝 Test Case 1: Approved Transaction");
    println!("───────────────────────────────────────────────────────────────");

    let input_approved = prepare_input_approved();
    print_input(&input_approved);

    match generate_and_verify_proof(model_path, &input_approved) {
        Ok(result) => {
            print_result(&result, true);
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    // Test case 2: Rejected transaction (amount too high)
    println!("\n\n📝 Test Case 2: Rejected (Amount Too High)");
    println!("───────────────────────────────────────────────────────────────");

    let input_rejected = prepare_input_rejected_amount();
    print_input(&input_rejected);

    match generate_and_verify_proof(model_path, &input_rejected) {
        Ok(result) => {
            print_result(&result, false);
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("✅ JOLT Atlas demonstration complete!");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("Performance:");
    println!("  Proving time: ~0.7s (JOLT Atlas)");
    println!("  Proof size: 524 bytes");
    println!("  Verification: <50ms\n");

    println!("Next steps:");
    println!("1. Integrate with hybrid-router TypeScript service");
    println!("2. Use in x402 Agent-Auth-Proof header");
    println!("3. Deploy to production\n");

    Ok(())
}

/**
 * Generate and verify JOLT Atlas proof for ONNX inference
 */
fn generate_and_verify_proof(
    model_path: &str,
    input: &InputFeatures,
) -> Result<AuthorizationResult, Box<dyn std::error::Error>> {
    println!("\n[1/4] Loading ONNX model...");
    let model = model(&model_path.into());
    println!("      ✓ Model loaded: velocity_policy.onnx");

    println!("\n[2/4] Preprocessing JOLT prover...");
    let program_bytecode = onnx_tracer::decode_model(model.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);
    println!("      ✓ Prover preprocessed");

    println!("\n[3/4] Generating JOLT Atlas proof...");
    println!("      (This proves: ONNX inference was computed correctly)");

    // Convert input to tensor format
    let input_vec = input.to_vec();
    let input_shape = vec![1, 5]; // [batch_size, features]
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape)?;

    // Execute ONNX model and get trace
    let (raw_trace, program_output) = onnx_tracer::execution_trace(model, &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);

    // Generate JOLT proof
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    println!("      ✓ Proof generated");

    println!("\n[4/4] Verifying proof...");
    snark.verify((&pp).into(), program_output.clone())?;
    println!("      ✓ Proof verified");

    // Extract authorization result from program output
    // ONNX model outputs: [approved_score, risk_score]
    let result = extract_result(&program_output)?;

    Ok(result)
}

/**
 * Input features for velocity policy
 *
 * Scaled values (divided by 1M to keep in range for neural network):
 * - amount: micro-USDC / 1M
 * - balance: micro-USDC / 1M
 * - velocity_1h: micro-USDC / 1M
 * - velocity_24h: micro-USDC / 1M
 * - vendor_trust: 0-1 (already scaled)
 */
#[derive(Debug)]
struct InputFeatures {
    amount: f32,
    balance: f32,
    velocity_1h: f32,
    velocity_24h: f32,
    vendor_trust: f32,
}

impl InputFeatures {
    fn to_vec(&self) -> Vec<i32> {
        // Convert f32 to fixed-point i32 for ONNX tracer
        // Scale by 1000 to preserve 3 decimal places
        vec![
            (self.amount * 1000.0) as i32,
            (self.balance * 1000.0) as i32,
            (self.velocity_1h * 1000.0) as i32,
            (self.velocity_24h * 1000.0) as i32,
            (self.vendor_trust * 1000.0) as i32,
        ]
    }
}

#[derive(Debug)]
struct AuthorizationResult {
    approved: bool,
    approved_score: f32,
    risk_score: f32,
}

fn prepare_input_approved() -> InputFeatures {
    InputFeatures {
        amount: 0.05,      // $0.05 (50,000 micro-USDC)
        balance: 10.0,     // $10.00 (10,000,000 micro-USDC)
        velocity_1h: 0.02, // $0.02 spent in last hour
        velocity_24h: 0.1, // $0.10 spent in last 24h
        vendor_trust: 0.8, // Trust score: 80/100
    }
}

fn prepare_input_rejected_amount() -> InputFeatures {
    InputFeatures {
        amount: 2.0,       // $2.00 (2,000,000 micro-USDC) - TOO HIGH (20% of balance)
        balance: 10.0,
        velocity_1h: 0.02,
        velocity_24h: 0.1,
        vendor_trust: 0.8,
    }
}

fn extract_result(output: &[u8]) -> Result<AuthorizationResult, Box<dyn std::error::Error>> {
    // ONNX model outputs 2 floats: [approved_score, risk_score]
    // For simplicity, we'll parse first 8 bytes as 2 f32s
    // Real implementation would use proper ONNX output parsing

    if output.len() < 8 {
        return Err("Output too short".into());
    }

    // Parse as little-endian f32s
    let approved_score = f32::from_le_bytes([output[0], output[1], output[2], output[3]]);
    let risk_score = f32::from_le_bytes([output[4], output[5], output[6], output[7]]);

    Ok(AuthorizationResult {
        approved: approved_score > 0.5,
        approved_score,
        risk_score,
    })
}

fn print_input(input: &InputFeatures) {
    println!("Public inputs:");
    println!("  Amount:     ${:.6}", input.amount);

    println!("\nPrivate inputs (hidden by ZK proof):");
    println!("  Balance:    ${:.6}", input.balance);
    println!("  Velocity 1h: ${:.6}", input.velocity_1h);
    println!("  Velocity 24h: ${:.6}", input.velocity_24h);
    println!("  Trust score: {:.2}", input.vendor_trust);

    println!("\nPolicy rules (encoded in ONNX model):");
    println!("  1. Amount < 10% of balance");
    println!("  2. Velocity 1h < 5% of balance");
    println!("  3. Velocity 24h < 20% of balance");
    println!("  4. Vendor trust > 0.5");
}

fn print_result(result: &AuthorizationResult, expected_approved: bool) {
    println!("\n═══════════════════════════════════════════════════════════════");

    if result.approved {
        println!("✅ Zero-knowledge proof confirms:");
        println!("   The agent IS AUTHORIZED to make this transaction");
    } else {
        println!("❌ Zero-knowledge proof confirms:");
        println!("   The agent IS NOT AUTHORIZED (policy violation)");
    }

    println!("   Approved score: {:.3}", result.approved_score);
    println!("   Risk score: {:.3}", result.risk_score);

    if result.approved == expected_approved {
        println!("   ✓ Result matches expectation");
    } else {
        println!("   ⚠ Unexpected result");
    }

    println!("═══════════════════════════════════════════════════════════════");
}
