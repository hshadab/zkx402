/*!
 * JOLT Atlas Prover for ZKx402 Agent Authorization
 *
 * This library provides zero-knowledge proof generation for ONNX-based
 * agent spending authorization policies using JOLT Atlas zkML.
 *
 * # Features
 * - Sub-second proof generation (~0.7s)
 * - Compact proofs (524 bytes)
 * - Privacy-preserving (hides balance, velocity, policy thresholds)
 * - ONNX model support (any ML framework)
 *
 * # Examples
 *
 * See `examples/simple_test.rs` for a minimal working example.
 */

// Re-export JOLT Atlas components for convenience
pub use zkml_jolt_core::{
    jolt::{JoltProverPreprocessing, JoltSNARK, execution_trace::jolt_execution_trace},
};

pub use onnx_tracer::{
    model,
    tensor::Tensor,
    ProgramIO,
    execution_trace,
    decode_model,
};

pub use ark_bn254::Fr;
pub use jolt_core::{
    poly::commitment::dory::DoryCommitmentScheme,
    utils::transcript::KeccakTranscript,
};

/// Type alias for the polynomial commitment scheme used by JOLT Atlas
pub type PCS = DoryCommitmentScheme<KeccakTranscript>;

/// Result type for this library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        // Basic sanity test
        assert_eq!(2 + 2, 4);
    }
}
