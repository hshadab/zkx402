# JOLT Atlas Blockers for Agent Authorization

## Summary

After extensive testing, **JOLT Atlas cannot currently support real-world agent authorization policies** due to severe operation limitations.

## Discovered Limitations

### Critical Operations Missing

1. **Greater (`>`) operation** - NOT SUPPORTED
   - Error: `Unknown op: >` at `utilities.rs:1374`
   - Only `<` (Less) and `>=` (Greater-or-Equal) are supported
   - This makes basic authorization rules impossible (e.g., "vendor_trust > 50")

2. **MatMult operations** - BROKEN
   - Assertion failures in `rebase_scale`
   - Prevents use of neural network models

3. **Type casting** - PARTIALLY SUPPORTED
   - Cast from Bool to I32 exists in ONNX spec
   - But triggers "Impossible to unify I32 with F32" errors
   - Suggests tract-onnx type inference issues

### What IS Supported

Based on `/home/hshadab/zkx402/zkx402-agent-auth/jolt-atlas-fork/onnx-tracer/src/graph/utilities.rs`:

**Supported operations (lines 1000-1376)**:
- Add, Sub, Mul (integer only)
- Less (`<`)
- GreaterEqual (`>=`)
- Reshape, Flatten
- Slice
- Constant
- Conv (with limitations)
- SumPool
- Pad
- **NOT** supported: Greater (`>`), Division, MatMult, Sigmoid, ReLU (with floats)

## Test Results

### Test 1: Integer-only ONNX with ReLU
**File**: `integer_rule_based_auth.onnx`
**Result**: ❌ FAILED
**Error**: `Failed analyse for node #59 "/Clip" Clip: Impossible to unify I32 with F32`
**Root cause**: torch.clamp creates F32 constants

### Test 2: Pure integer arithmetic (no ReLU)
**File**: `pure_integer_auth.onnx`
**Result**: ❌ FAILED
**Error**: `Unknown op: >`
**Root cause**: PyTorch exports comparison as "Greater" operation, which JOLT Atlas doesn't support

### Test 3: Simple Identity function
**File**: `simple_identity.onnx`
**Result**: ✅ SUCCESS
**Proof time**: ~0.7 seconds
**Note**: Only works because it has NO authorization logic

## Authorization Requirements vs JOLT Atlas Capabilities

### What Authorization Needs:
```
✓ Amount < 10% of balance  (uses <, SUPPORTED)
✗ Vendor trust > 0.5       (uses >, NOT SUPPORTED)
✗ Score >= threshold       (uses >=, SUPPORTED)
✗ Neural network scoring   (uses MatMult, BROKEN)
```

### Workaround Attempts:

1. **Use `>=` instead of `>`**:
   - Original: `vendor_trust > 50`
   - Workaround: `vendor_trust >= 51`
   - **Status**: Theoretically possible, but very hacky

2. **Avoid comparisons entirely**:
   - Use arithmetic only (Add, Sub, Mul)
   - **Problem**: Cannot make binary decisions (approve/reject)

3. **Implement Greater operation in JOLT Atlas**:
   - Would require modifying jolt-atlas-fork
   - Not trivial - needs zkVM instruction support

## Impact on X402 Integration

### Blocker #1: No auth service needed ✅ RESOLVED
- User clarified: X402 is header-based, not service-based
- Proofs generated CLIENT-SIDE
- Sent via `X-Agent-Auth-Proof` header

### Blocker #2: Integer-only ONNX models ❌ BLOCKED
- JOLT Atlas supports integers
- But missing critical operations (Greater, Division)
- **Cannot implement real authorization logic**

### Blocker #3: Proof format ⏸️ BLOCKED BY #2
- Can't define format until we have working proofs

## Recommended Next Steps

### Option 1: Extend JOLT Atlas (High Effort)
1. Add Greater (`>`) operation to `utilities.rs`
2. Implement zkVM instruction for comparison
3. Fix MatMult rebase_scale issues
4. Estimated time: 2-4 weeks

### Option 2: Use Different zkML System (Medium Effort)
1. Evaluate alternatives:
   - EZKL (more mature, broader ONNX support)
   - Risc0-zkVM (full VM, not just ML)
   - Noir (custom circuits)
2. Migrate codebase
3. Estimated time: 1-2 weeks

### Option 3: Simplify Authorization (Low Effort)
1. Use ONLY supported operations (`<`, `>=`, Add, Sub, Mul)
2. Workaround missing `>` with `>= threshold+1`
3. No neural networks, only arithmetic rules
4. **Trade-off**: Very limited authorization capabilities
5. Estimated time: 2-3 days

## Files Created

- `/home/hshadab/zkx402/zkx402-agent-auth/policy-examples/onnx/create_integer_policy.py` - Integer policy (failed: Clip/F32 issue)
- `/home/hshadab/zkx402/zkx402-agent-auth/policy-examples/onnx/create_pure_integer_policy.py` - Pure integer (failed: Missing Greater op)
- `/home/hshadab/zkx402/zkx402-agent-auth/jolt-prover/examples/integer_auth_e2e.rs` - E2E test (never passed)

## Conclusion

**JOLT Atlas is not production-ready for agent authorization.**

The missing `>` (Greater) operation is a fundamental blocker. While workarounds exist (using `>=` with adjusted thresholds), the system's limitations make it unsuitable for real-world authorization policies that require:
- Complex decision logic
- Neural network scoring
- Standard comparison operations

**Recommendation**: Evaluate EZKL or other zkML systems that have more complete ONNX operation support.
