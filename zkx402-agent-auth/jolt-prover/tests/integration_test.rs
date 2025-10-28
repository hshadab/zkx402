/**
 * Integration tests for JOLT Atlas proof generation
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
use std::path::Path;

type PCS = DoryCommitmentScheme<KeccakTranscript>;

#[test]
fn test_simple_auth_approved() {
    // Skip if model doesn't exist (CI environment)
    let model_path = "../policy-examples/onnx/simple_auth.onnx";
    if !Path::new(model_path).exists() {
        println!("Skipping test: model not found");
        return;
    }

    // Load model
    let model_obj = model(&model_path.into());
    let program_bytecode = onnx_tracer::decode_model(model_obj.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);

    // Test case: should approve
    // Amount: 5 ($0.05), Balance: 1000 ($10.00), Trust: 80 (0.80)
    let input_vec = vec![5i32, 1000i32, 2i32, 10i32, 80i32];
    let input_shape = vec![1, 5];
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape).unwrap();

    // Generate proof
    let (raw_trace, program_output) = onnx_tracer::execution_trace(model_obj.clone(), &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    // Verify proof
    let is_valid = snark.verify((&pp).into(), program_output.clone()).is_ok();
    assert!(is_valid, "Proof verification should succeed");

    // Check authorization result
    let approved_score = program_output.output.inner[0];
    assert!(approved_score > 50, "Transaction should be approved");
}

#[test]
fn test_simple_auth_rejected_excessive_amount() {
    let model_path = "../policy-examples/onnx/simple_auth.onnx";
    if !Path::new(model_path).exists() {
        println!("Skipping test: model not found");
        return;
    }

    let model_obj = model(&model_path.into());
    let program_bytecode = onnx_tracer::decode_model(model_obj.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);

    // Test case: should reject (amount > 10% balance)
    // Amount: 200 ($2.00), Balance: 1000 ($10.00)
    let input_vec = vec![200i32, 1000i32, 2i32, 10i32, 80i32];
    let input_shape = vec![1, 5];
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape).unwrap();

    let (raw_trace, program_output) = onnx_tracer::execution_trace(model_obj.clone(), &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    let is_valid = snark.verify((&pp).into(), program_output.clone()).is_ok();
    assert!(is_valid, "Proof verification should succeed even for rejected transactions");

    let approved_score = program_output.output.inner[0];
    assert!(approved_score <= 50, "Transaction should be rejected");
}

#[test]
fn test_simple_auth_rejected_low_trust() {
    let model_path = "../policy-examples/onnx/simple_auth.onnx";
    if !Path::new(model_path).exists() {
        println!("Skipping test: model not found");
        return;
    }

    let model_obj = model(&model_path.into());
    let program_bytecode = onnx_tracer::decode_model(model_obj.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);

    // Test case: should reject (trust < 50)
    // Trust: 30 (0.30)
    let input_vec = vec![5i32, 1000i32, 2i32, 10i32, 30i32];
    let input_shape = vec![1, 5];
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape).unwrap();

    let (raw_trace, program_output) = onnx_tracer::execution_trace(model_obj.clone(), &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    let is_valid = snark.verify((&pp).into(), program_output.clone()).is_ok();
    assert!(is_valid);

    let approved_score = program_output.output.inner[0];
    assert!(approved_score <= 50, "Transaction should be rejected due to low trust");
}

#[test]
fn test_neural_auth() {
    let model_path = "../policy-examples/onnx/neural_auth.onnx";
    if !Path::new(model_path).exists() {
        println!("Skipping test: model not found");
        return;
    }

    let model_obj = model(&model_path.into());
    let program_bytecode = onnx_tracer::decode_model(model_obj.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);

    // Test with reasonable values
    let input_vec = vec![50i32, 1000i32, 20i32, 100i32, 75i32];
    let input_shape = vec![1, 5];
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape).unwrap();

    let (raw_trace, program_output) = onnx_tracer::execution_trace(model_obj.clone(), &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);

    // Just verify it completes without panicking
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);

    let is_valid = snark.verify((&pp).into(), program_output.clone()).is_ok();
    assert!(is_valid, "Neural network proof should verify correctly");
}

#[test]
fn test_proof_performance() {
    let model_path = "../policy-examples/onnx/simple_auth.onnx";
    if !Path::new(model_path).exists() {
        println!("Skipping test: model not found");
        return;
    }

    let model_obj = model(&model_path.into());
    let program_bytecode = onnx_tracer::decode_model(model_obj.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);

    let input_vec = vec![5i32, 1000i32, 2i32, 10i32, 80i32];
    let input_shape = vec![1, 5];
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape).unwrap();

    let (raw_trace, program_output) = onnx_tracer::execution_trace(model_obj.clone(), &input_tensor);
    let execution_trace = jolt_execution_trace(raw_trace);

    // Measure proof generation time
    let start = std::time::Instant::now();
    let snark: JoltSNARK<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prove(pp.clone(), execution_trace, &program_output);
    let proving_time = start.elapsed();

    println!("Proving time: {:?}", proving_time);

    // Measure verification time
    let start = std::time::Instant::now();
    let is_valid = snark.verify((&pp).into(), program_output.clone()).is_ok();
    let verify_time = start.elapsed();

    println!("Verification time: {:?}", verify_time);

    assert!(is_valid);
    assert!(proving_time.as_secs() < 5, "Proving should complete within 5 seconds");
    assert!(verify_time.as_millis() < 500, "Verification should complete within 500ms");
}
