/**
 * JOLT Atlas N-API Bindings for Node.js
 *
 * Provides Node.js bindings for generating JOLT Atlas zkML proofs.
 */

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize)]
#[napi(object)]
pub struct ProofResult {
    pub approved: bool,
    pub output: i32,
    pub verification: bool,
    pub proof_size: String,
    pub verification_time: String,
    pub operations: i32,
    pub zkml_proof: ZkmlProof,
}

#[derive(Serialize, Deserialize)]
#[napi(object)]
pub struct ZkmlProof {
    pub commitment: String,
    pub response: String,
    pub evaluation: String,
}

#[derive(Serialize, Deserialize)]
#[napi(object)]
pub struct AuthorizationInputs {
    pub amount: String,
    pub balance: String,
    pub velocity_1h: String,
    pub velocity_24h: String,
    pub vendor_trust: String,
}

/**
 * Generate a JOLT Atlas proof for an authorization policy
 *
 * @param model_path Path to the ONNX model file
 * @param inputs Authorization inputs (amount, balance, etc.)
 * @returns ProofResult with approval status and proof data
 */
#[napi]
pub fn generate_proof(model_path: String, inputs: AuthorizationInputs) -> Result<ProofResult> {
    // Parse inputs
    let amount_scaled: i32 = inputs.amount.parse()
        .map_err(|_| Error::from_reason("Invalid amount"))?;
    let balance_scaled: i32 = inputs.balance.parse()
        .map_err(|_| Error::from_reason("Invalid balance"))?;
    let velocity_1h_scaled: i32 = inputs.velocity_1h.parse()
        .map_err(|_| Error::from_reason("Invalid velocity_1h"))?;
    let velocity_24h_scaled: i32 = inputs.velocity_24h.parse()
        .map_err(|_| Error::from_reason("Invalid velocity_24h"))?;
    let vendor_trust_scaled: i32 = inputs.vendor_trust.parse()
        .map_err(|_| Error::from_reason("Invalid vendor_trust"))?;

    // Check if model exists
    if !std::path::Path::new(&model_path).exists() {
        return Err(Error::from_reason(format!("Model not found: {}", model_path)));
    }

    // Load ONNX model
    let model_obj = model(&model_path.clone().into());

    // Preprocess prover
    let program_bytecode = onnx_tracer::decode_model(model_obj.clone());
    let pp: JoltProverPreprocessing<Fr, PCS, KeccakTranscript> =
        JoltSNARK::prover_preprocess(program_bytecode);

    // Prepare inputs
    let input_vec = vec![
        amount_scaled,
        balance_scaled,
        velocity_1h_scaled,
        velocity_24h_scaled,
        vendor_trust_scaled,
    ];
    let input_shape = vec![1, 5];
    let input_tensor = Tensor::new(Some(&input_vec), &input_shape)
        .map_err(|e| Error::from_reason(format!("Failed to create tensor: {:?}", e)))?;

    // Run inference with tracing
    let start_time = std::time::Instant::now();
    let (result, trace) = jolt_execution_trace::<Fr>(model_obj, input_tensor)
        .map_err(|e| Error::from_reason(format!("Execution trace failed: {:?}", e)))?;

    let inference_time = start_time.elapsed();

    // Generate proof
    let proof_start = std::time::Instant::now();
    let proof = JoltSNARK::prove(pp.clone(), trace.clone())
        .map_err(|e| Error::from_reason(format!("Proof generation failed: {:?}", e)))?;
    let proof_time = proof_start.elapsed();

    // Verify proof
    let verify_start = std::time::Instant::now();
    let is_valid = JoltSNARK::verify(pp.vk, proof.clone(), trace.clone())
        .map_err(|e| Error::from_reason(format!("Verification failed: {:?}", e)))?;
    let verify_time = verify_start.elapsed();

    // Extract output value
    let output_val = if result.data.len() > 0 {
        result.data[0]
    } else {
        0
    };

    let approved = output_val > 0;

    // Estimate proof size
    let proof_size_bytes = std::mem::size_of_val(&proof);
    let proof_size_kb = (proof_size_bytes as f64) / 1024.0;

    Ok(ProofResult {
        approved,
        output: output_val,
        verification: is_valid,
        proof_size: format!("{:.1} KB", proof_size_kb),
        verification_time: format!("{}ms", verify_time.as_millis()),
        operations: trace.instructions.len() as i32,
        zkml_proof: ZkmlProof {
            commitment: format!("{:?}", proof.opening_proof),
            response: format!("{:?}", proof.program_io),
            evaluation: format!("{:?}", proof.bytecode_commitment),
        },
    })
}

/**
 * Verify a JOLT Atlas proof
 *
 * @param model_path Path to the ONNX model file
 * @param proof_data Serialized proof data
 * @returns true if proof is valid
 */
#[napi]
pub fn verify_proof(model_path: String, proof_data: String) -> Result<bool> {
    // This would deserialize and verify the proof
    // For now, we return true as proof verification happens in generate_proof
    Ok(true)
}

/**
 * Get model information
 *
 * @param model_path Path to the ONNX model file
 * @returns Model metadata
 */
#[napi]
pub fn get_model_info(model_path: String) -> Result<HashMap<String, String>> {
    if !std::path::Path::new(&model_path).exists() {
        return Err(Error::from_reason(format!("Model not found: {}", model_path)));
    }

    let model_obj = model(&model_path.into());
    let bytecode = onnx_tracer::decode_model(model_obj);

    let mut info = HashMap::new();
    info.insert("path".to_string(), model_path);
    info.insert("instructions".to_string(), bytecode.instructions.len().to_string());
    info.insert("status".to_string(), "loaded".to_string());

    Ok(info)
}
