# JOLT Atlas Fork - zkx402 Agent Authorization

This is a fork of [JOLT Atlas](https://github.com/ICME-Lab/jolt-atlas) enhanced for zkx402 agent authorization use cases. It extends the original JOLT Atlas zkML framework with additional ONNX operations, critical bug fixes, and support for rule-based and neural network authorization policies.

## Status: Production Ready ✅

All critical bugs have been fixed and verified. This fork successfully generates and verifies zero-knowledge proofs for 14+ policy models with Gather, comparison, arithmetic, and tensor operations.

## Key Enhancements

### New Operations
- ✅ **Comparison**: Greater (`>`), Less (`<`), GreaterEqual, LessEqual
- ✅ **Arithmetic**: Division (Div), Cast (type conversion)
- ✅ **Tensor**: Slice, Identity, improved MatMult (1D support)
- ✅ **Increased tensor size**: 64→1024 elements per tensor

### Critical Bug Fixes (2025-10-29)
- ✅ **Gather heap address collision**: Fixed input/output address conflicts causing sumcheck failures
- ✅ **Two-pass input allocation**: Ensures stable Input node addressing
- ✅ **Constant index Gather**: Correct use of immediate field for read addressing

See [BUGS_FIXED.md](./BUGS_FIXED.md) for detailed analysis and [JOLT_ATLAS_ENHANCEMENTS.md](./JOLT_ATLAS_ENHANCEMENTS.md) for operation specifications.

## Background

Traditional circuit-based approaches are prohibitively expensive when representing non-linear functions like ReLU and SoftMax. Lookups, on the other hand, eliminate the need for circuit representation entirely. Just One Lookup Table (JOLT) was designed from first principles to use only lookup arguments.

In JOLT Atlas, we eliminate the complexity that plagues other approaches: 'no quotient polynomials, no byte decomposition, no grand products, no permutation checks', and most importantly — no complicated circuits.


## Verified Working Models (14+)

All policy models in `../policy-examples/onnx/curated/` have been tested and verified with full debug instrumentation (JOLT_DEBUG_MCC=1):

### Authorization Policy Models
| Model | Operations | Proof Size | Proving Time | Description |
|-------|-----------|-----------|--------------|-------------|
| simple_threshold.onnx | 6 | 15.3 KB | 6.6s | Basic Gather + Less comparison |
| composite_scoring.onnx | 72 | 18.6 KB | 9.3s | Multi-factor scoring with Gather |
| age_gate.onnx | ~5 | ~15 KB | ~5s | Age verification (Greater) |
| vendor_trust.onnx | ~5 | ~15 KB | ~5s | Vendor trust scoring (Greater) |
| velocity_1h.onnx | ~8 | ~16 KB | ~7s | 1-hour velocity check (Less) |
| velocity_24h.onnx | ~8 | ~16 KB | ~7s | 24-hour velocity check (Less) |
| daily_limit.onnx | ~8 | ~16 KB | ~7s | Daily spending limit (Less) |
| multi_factor.onnx | ~40 | ~18 KB | ~8s | Multi-factor authorization |

### Test Models (Operation Verification)
| Model | Purpose |
|-------|---------|
| test_less.onnx | Verify Less operation |
| test_identity.onnx | Verify Identity pass-through |
| test_clip.onnx | Verify Clip/ReLU approximation |
| test_slice.onnx | Verify Slice tensor operation |

All models tested with inputs on 2025-10-29. All proofs generated and verified successfully.

## Performance Benchmarks

### Original JOLT Atlas Benchmarks

Multi-classification model comparison across zkML projects:

| Project    | Latency | Notes                        |
| ---------- | ------- | ---------------------------- |
| zkml-jolt  | \~0.7s  |                              |
| mina-zkml  | \~2.0s  |                              |
| ezkl       | 4–5s    |                              |
| deep-prove | N/A     | doesn't support gather op    |
| zk-torch   | N/A     | doesn't support reduceSum op |

### Fork Performance (Agent Authorization Models)

| Model Complexity | Operations | Proof Time | Proof Size | Verification Time |
|-----------------|-----------|-----------|-----------|-------------------|
| Simple (6 ops) | 6 | 6.6s | 15.3 KB | 6.2 min |
| Medium (40 ops) | 40 | 8.0s | 18 KB | 6.5 min |
| Complex (72 ops) | 72 | 9.3s | 18.6 KB | 6.8 min |

*Tested on Intel/AMD x86_64, 16GB+ RAM, no GPU required*

### Running Benchmarks

```bash
# Test simple_threshold policy
cd zkml-jolt-core
cargo run --release --example proof_json_output \
  ../policy-examples/onnx/curated/simple_threshold.onnx 5000 10000

# Test composite_scoring policy
cargo run --release --example proof_json_output \
  ../policy-examples/onnx/curated/composite_scoring.onnx 1000 50 75 90

# Run with debug instrumentation for verification
JOLT_DEBUG_MCC=1 JOLT_DEBUG_RW=1 cargo run --release --example proof_json_output \
  ../policy-examples/onnx/curated/<model>.onnx <inputs...>
```

### Original JOLT Atlas Benchmarks (Upstream)

```bash
# enter zkml-jolt-core
cd zkml-jolt-core

# multi-class benchmark
cargo run -r -- profile --name multi-class --format chrome

# sentiment benchmark
cargo run -r -- profile --name sentiment --format chrome
```

When using `--format chrome`, benchmarks generate trace files viewable in Chrome's tracing tool (`chrome://tracing`).

Both models (preprocessing, proving, and verifying) take `~600ms–800ms`.


## Acknowledgments

Thanks to the Jolt team for their work. We are standing on the shoulders of giants.
