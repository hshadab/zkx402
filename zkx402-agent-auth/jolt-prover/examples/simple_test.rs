/**
 * Minimal JOLT Atlas ONNX Test
 *
 * This tests the simplest possible ONNX model: Identity function (y = x)
 * No matrix multiplication, no complex operations - just pass through.
 *
 * If this works, we know JOLT Atlas basics are functional.
 */

use zkml_jolt_core::{
    jolt::{JoltProverPreprocessing, JoltSNARK, execution_trace::jolt_execution_trace},
};
use ark_bn254::Fr;
use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme,
    utils::transcript::KeccakTranscript,
};
use onnx_tracer::{model, tensor::Tensor, ProgramIO};

type PCS = DoryCommitmentScheme<KeccakTranscript>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  JOLT Atlas Minimal ONNX Test - Identity Function    ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Check if ONNX model exists
    let model_path = "../policy-examples/onnx/simple_test.onnx";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("❌ ONNX model not found at: {}", model_path);
        eprintln!("\nPlease create the model first:");
        eprintln!("  cd ../policy-examples/onnx");
        eprintln!("  python3 train_simple.py\n");
        return Ok(());
    }

    println!("[1/5] Loading ONNX model...");
    let model = model(&model_path.into());
    println!("      ✓ Model loaded: simple_test.onnx (Identity function)");

    println!("\n[2/5] Preprocessing JOLT prover...");
    let program_bytecode = onnx_tracer::decode_model(model.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);
    println!("      ✓ Prover preprocessed");

    println!("\n[3/5] Preparing test input...");
    // Simple test: input = 42
    let input_value = 42i32;
    let input_vec = vec![input_value];
    let input_shape = vec![1, 1]; // [batch_size, features]
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape)?;
    println!("      ✓ Input prepared: {}", input_value);

    println!("\n[4/5] Generating JOLT Atlas proof...");
    println!("      (This proves: Identity function computed correctly)");

    // Execute ONNX model and get trace
    let (raw_trace, program_output) = onnx_tracer::execution_trace(model, &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);

    // Generate JOLT proof
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    println!("      ✓ Proof generated");

    println!("\n[5/5] Verifying proof...");
    snark.verify((&pp).into(), program_output.clone())?;
    println!("      ✓ Proof verified!");

    // Extract output
    let output_value = program_output.output.inner[0];

    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║                    TEST RESULT                        ║");
    println!("╠═══════════════════════════════════════════════════════╣");
    println!("║  Input:  {:>43} ║", input_value);
    println!("║  Output: {:>43} ║", output_value);
    println!("║  Match:  {:>43} ║", if input_value == output_value { "✓ YES" } else { "✗ NO" });
    println!("╠═══════════════════════════════════════════════════════╣");
    println!("║  ✅ JOLT Atlas ONNX Proof Generation: SUCCESS       ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    println!("Next steps:");
    println!("1. Try more complex ONNX models");
    println!("2. Add actual authorization logic");
    println!("3. Integrate with X402 protocol\n");

    Ok(())
}
