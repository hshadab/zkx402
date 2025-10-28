/**
 * Simple End-to-End JOLT Atlas Proving Example
 *
 * This demonstrates the complete JOLT Atlas proving pipeline:
 * 1. Load ONNX model
 * 2. Prepare input tensor
 * 3. Generate execution trace
 * 4. Preprocess prover
 * 5. Generate JOLT proof
 * 6. Verify proof
 *
 * Uses a minimal 5→8→2 velocity model (~66 elements)
 */

use ark_bn254::Fr;
use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme,
    utils::transcript::KeccakTranscript,
};
use onnx_tracer::{model, tensor::Tensor};
use zkml_jolt_core::jolt::{
    execution_trace::jolt_execution_trace, JoltProverPreprocessing, JoltSNARK,
};

type PCS = DoryCommitmentScheme<KeccakTranscript>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(60));
    println!("JOLT Atlas End-to-End Proving Example");
    println!("{}", "=".repeat(60));

    // Check if model exists
    let model_path = "../policy-examples/onnx/simple_velocity_policy_no_gemm.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("\n❌ ONNX model not found at: {}", model_path);
        eprintln!("\nPlease train the model first:");
        eprintln!("  cd ../policy-examples/onnx");
        eprintln!("  python3 train_simple_velocity.py\n");
        return Ok(());
    }

    println!("\nModel: {}", model_path);
    println!("Architecture: 5 → 8 → 2 (~ 66 parameters)");

    // Test input: [amount, balance, velocity_1h, velocity_24h, vendor_trust]
    // All values normalized to 0-1 range
    let test_input = vec![
        50,   // amount: 0.05 (5% of balance)
        800,  // balance: 0.80
        10,   // velocity_1h: 0.01
        50,   // velocity_24h: 0.05
        700,  // vendor_trust: 0.70
    ];

    println!("\nTest Input (scaled to 0-1000 range):");
    println!("  Amount:       {} (0.{:03})", test_input[0], test_input[0]);
    println!("  Balance:      {} (0.{:03})", test_input[1], test_input[1]);
    println!("  Velocity 1h:  {} (0.{:03})", test_input[2], test_input[2]);
    println!("  Velocity 24h: {} (0.{:03})", test_input[3], test_input[3]);
    println!("  Vendor trust: {} (0.{:03})", test_input[4], test_input[4]);

    println!("\n{}", "=".repeat(60));
    println!("JOLT Atlas Proving Pipeline");
    println!("{}", "=".repeat(60));

    // Step 1: Load ONNX model
    println!("\n[1/6] Loading ONNX model...");
    let onnx_model = model(&model_path.into());
    println!("      ✓ Model loaded successfully");

    // Step 2: Decode to JOLT bytecode
    println!("\n[2/6] Decoding ONNX to JOLT bytecode...");
    let program_bytecode = onnx_tracer::decode_model(onnx_model.clone());
    println!("      ✓ Bytecode generated");

    // Step 3: Preprocess prover (expensive, done once per model)
    println!("\n[3/6] Preprocessing prover...");
    println!("      (This is expensive but only done once per model)");
    let start = std::time::Instant::now();
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);
    let preprocess_time = start.elapsed();
    println!("      ✓ Prover preprocessed in {:.2?}", preprocess_time);

    // Step 4: Prepare input tensor
    println!("\n[4/6] Preparing input tensor...");
    let input_shape = vec![1, 5]; // [batch_size=1, features=5]
    let input_tensor = Tensor::new(Some(&test_input), &input_shape)?;
    println!("      ✓ Input tensor shape: {:?}", input_shape);
    println!("      ✓ Input tensor data: {:?}", test_input);

    // Step 5: Generate execution trace
    println!("\n[5/6] Executing ONNX model and generating trace...");
    let (raw_trace, program_output) =
        onnx_tracer::execution_trace(onnx_model, &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);
    println!("      ✓ Execution trace generated");
    println!("      ✓ Program output dims: {:?}", program_output.output.dims());
    println!("      ✓ Program output data length: {} elements", program_output.output.inner.len());

    // Step 6: Generate JOLT proof
    println!("\n[6/6] Generating JOLT proof...");
    println!("      (This is the expensive step - proving ONNX inference)");
    let start = std::time::Instant::now();
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);
    let prove_time = start.elapsed();
    println!("      ✓ Proof generated in {:.2?}", prove_time);

    // Step 7: Verify proof
    println!("\n[7/7] Verifying proof...");
    let start = std::time::Instant::now();
    snark.verify((&pp).into(), program_output.clone())?;
    let verify_time = start.elapsed();
    println!("      ✓ Proof verified in {:.2?}", verify_time);

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("✅ End-to-End JOLT Proving Complete!");
    println!("{}", "=".repeat(60));

    println!("\nPerformance Summary:");
    println!("  Preprocessing: {:.2?} (one-time cost)", preprocess_time);
    println!("  Proving:       {:.2?} (per-request)", prove_time);
    println!("  Verification:  {:.2?} (per-request)", verify_time);
    println!("  Total:         {:.2?}", preprocess_time + prove_time + verify_time);

    println!("\nModel Details:");
    println!("  Parameters: ~66");
    println!("  Elements: ~66 (well under MAX_TENSOR_SIZE)");
    println!("  Layers: 3 (Linear, ReLU, Linear, Sigmoid)");

    println!("\nNext Steps:");
    println!("  1. ✅ Simple model works");
    println!("  2. ⏳ Test whitelist model (408 elements, requires MAX_TENSOR_SIZE=1024)");
    println!("  3. ⏳ Test business hours model (140 elements)");
    println!("  4. ⏳ Benchmark all transformed policies");

    Ok(())
}
