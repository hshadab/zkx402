# JOLT Atlas Build Issues - Deep Analysis

**Status**: ❌ Build failing despite excellent user fixes

**Date**: 2025-10-27

---

## Summary

Despite excellent fixes to resolve git submodule hangs and cargo lock contention, the build still fails with ark_bn254 version conflicts in the `dory` library.

---

## Root Cause

The issue is a **double-dependency** on arkworks algebra library:

```
jolt-core (ICME-Lab/zkml-jolt)
  ├── dory → ark-bn254 = "0.5.0" (patched to a16z fork → resolves to 0.5.0-alpha.0)
  └── dory → jolt-optimizations → ark-bn254 = "0.5.0-alpha.0" (from a16z fork directly)
```

**Result**: Two different versions of the same library with incompatible types.

---

## Dependency Tree Analysis

```bash
$ cargo tree -i ark-bn254

# Version 1: 0.5.0-alpha.0 (from a16z fork directly)
ark-bn254 v0.5.0-alpha.0 (https://github.com/a16z/arkworks-algebra?branch=dev%2Ftwist-shout#89de8361)
└── jolt-optimizations v0.5.0 (https://github.com/a16z/arkworks-algebra?branch=dev%2Ftwist-shout#89de8361)
    └── dory v1.0.0 (https://github.com/spaceandtimefdn/sxt-dory?branch=dev%2Ftwist-shout#97e92ae9)
        └── jolt-core v0.1.0 (https://github.com/ICME-Lab/zkml-jolt?branch=zkml-jolt#bb304ef0)

# Version 2: 0.5.0 (from crates.io, patched to a16z fork but resolved as 0.5.0)
ark-bn254 v0.5.0
├── dory v1.0.0 (https://github.com/spaceandtimefdn/sxt-dory?branch=dev%2Ftwist-shout#97e92ae9)
├── jolt-core v0.1.0 (https://github.com/ICME-Lab/zkml-jolt?branch=zkml-jolt#bb304ef0)
├── zkml-jolt-core v0.1.0 (local fork)
└── zkx402-jolt-auth v0.1.0 (our project)
```

---

## Error Messages

```
error[E0308]: mismatched types
  --> /home/hshadab/.cargo/git/checkouts/sxt-dory-ebc7f7f27b088cd2/97e92ae/src/curve.rs:1260:13
     |
1260 |             &addends_proj,
     |             ^^^^^^^^^^^^^ expected `Projective<Config>`, found `Projective<Config>`
     |
     = note: `Projective<Config>` and `Projective<Config>` have similar names, but are actually distinct types
note: `Projective<Config>` is defined in crate `ark_ec`
  --> /home/hshadab/.cargo/git/checkouts/arkworks-algebra-55219ebe4db9d51c/89de836/ec/src/models/short_weierstrass/mod.rs:8:1
     |
   8 | pub struct Projective<P: SWCurveConfig> {
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
note: `Projective<Config>` is defined in crate `ark_ec`
  --> /home/hshadab/.cargo/git/checkouts/arkworks-algebra-55219ebe4db9d51c/89de836/ec/src/models/short_weierstrass/mod.rs:8:1
     |
   8 | pub struct Projective<P: SWCurveConfig> {
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     = note: perhaps two different versions of crate `ark_ec` are being used?

error[E0308]: mismatched types
     |
     = note: expected reference `&[ark_ec::short_weierstrass::Projective<ark_bn254::curves::g2::Config>]`
                found reference `&Vec<ark_ec::short_weierstrass::Projective<ark_bn254::g2::Config>>`
```

**Key insight**: `ark_bn254::curves::g2::Config` vs `ark_bn254::g2::Config` are from two different versions!

---

## User's Excellent Fixes (That Worked Partially)

### Fix 1: `.cargo/config.toml` ✅
```toml
[registries.crates-io]
protocol = "sparse"

[net]
git-fetch-with-cli = true
retry = 3
```

**Impact**:
- ✅ Eliminated git submodule hangs
- ✅ Faster crates.io index updates
- ✅ Better network reliability

### Fix 2: `Cargo.toml` Dependency Unification ✅
```toml
[dependencies]
# Align jolt-core to same fork
jolt-core = { git = "https://github.com/ICME-Lab/zkml-jolt", package = "jolt-core", branch = "zkml-jolt", default-features = false }

# Unified arkworks to 0.5
ark-bn254 = "0.5"
ark-ff = "0.5"

[patch.crates-io]
# Force single arkworks source
ark-ff = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-ec = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-serialize = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-bn254 = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
```

**Impact**:
- ✅ Build progressed much further (~200 crates compiled)
- ✅ Reduced cache lock contention
- ❌ Still fails on dory library type mismatches

---

## Why The Fixes Didn't Fully Resolve It

The patches correctly redirect crates.io dependencies to the a16z fork, BUT:

1. **Cargo's patch resolution is conservative**: When it patches `ark-bn254 = "0.5.0"` to the a16z fork, it resolves to version `0.5.0` (even though the fork publishes as `0.5.0-alpha.0`)

2. **Direct git dependencies bypass patches**: `jolt-optimizations` is part of the arkworks fork and directly declares `ark-bn254 = "0.5.0-alpha.0"`, which cargo treats as a DIFFERENT package than `ark-bn254 = "0.5.0"`

3. **dory library depends on both paths**:
   - Direct: `ark-bn254 = "0.5.0"` (patched)
   - Indirect via jolt-optimizations: `ark-bn254 = "0.5.0-alpha.0"` (from fork)

---

## Attempted Solution That Cargo Won't Allow

We can't add this to our Cargo.toml because you can't patch one version to another version of the same package:

```toml
[patch.crates-io]
# This doesn't work - cargo won't let you patch to a different version
ark-bn254 = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout", version = "=0.5.0-alpha.0" }
```

---

## Potential Solutions

### Solution 1: Fork and Fix dory Library ⚠️ (High effort)

Fork `https://github.com/spaceandtimefdn/sxt-dory` and change all arkworks dependencies to `0.5.0-alpha.0`:

```toml
# In dory/Cargo.toml
[dependencies]
ark-bn254 = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-ec = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
```

Then patch dory in our Cargo.toml to use our fork.

**Pros**:
- Complete control
- Guaranteed to work

**Cons**:
- High maintenance burden
- Need to keep fork in sync
- Complex dependency on forks-of-forks

---

### Solution 2: Wait for Upstream Fix 🕐 (Low effort, uncertain timeline)

The ICME-Lab/zkml-jolt project likely knows about this issue. It may be resolved in future commits.

**Pros**:
- Zero effort
- Proper long-term solution

**Cons**:
- Unknown timeline
- May never be fixed if project is abandoned
- Blocks our progress

---

### Solution 3: Use JOLT Core Directly Without zkml-jolt-core ⚡ (Medium effort)

Instead of using the ICME-Lab fork, we could:
1. Use a16z/jolt directly (stable version)
2. Manually implement ONNX → JOLT bytecode transformation
3. Lose the MAX_TENSOR_SIZE=1024 optimization

**Pros**:
- No version conflicts
- More stable codebase

**Cons**:
- Lose our MAX_TENSOR_SIZE=1024 fork (critical for large models!)
- Have to reimplement onnx-tracer functionality
- Much more work

---

### Solution 4: Downgrade to Original JOLT Atlas (MAX_TENSOR_SIZE=64) ⚠️ (Quick but limiting)

Use the original unmodified JOLT Atlas with MAX_TENSOR_SIZE=64.

**Pros**:
- Might build cleanly
- Fast to try

**Cons**:
- ❌ **Can't prove budget_limits policy** (1,217 params → too large)
- ❌ **Can't prove merchant_limits policy** (1,377 params → too large)
- ❌ **Can't prove whitelist policy** (8,705 params → WAY too large)
- ✅ Can only prove velocity policy (~200 params)
- **Defeats the entire purpose of our fork!**

---

### Solution 5: Ship Without Proofs (zkML Models Only) 🚀 (RECOMMENDED FOR NOW)

Since we've already decided to ship zkML-only (no zkEngine), we could:

1. ✅ Keep all our trained ONNX models (already done, 98-100% accuracy)
2. ✅ Keep all our transformation framework (already documented)
3. ✅ Build the authorization API without proofs initially
4. ✅ Add JOLT proving later when build issues are resolved

**Validation approach**:
- Test ONNX models directly (without ZK proofs)
- Validate accuracy on test datasets
- Ship the authorization logic
- Add ZK proving in v2 when build stable

**Pros**:
- ✅ Unblocks shipping
- ✅ All critical work is done (policies + models)
- ✅ Can still add proofs later
- ✅ Users can test authorization logic immediately

**Cons**:
- ❌ No privacy (no ZK proofs)
- ❌ Not the full zkML vision

**Why this makes sense**:
- Our zkML-only architecture is complete and documented
- All 6 policies are trained with 98-100% accuracy
- ONNX models are exported and validated
- Authorization logic can be tested without proofs
- Proving can be added in Phase 2 when build stable

---

## Recommendation

**Phase 1 (NOW)**: Ship zkML-only without proofs
- Build authorization API
- Use ONNX models directly (no ZK)
- Validate all policy logic
- Document limitations

**Phase 2 (LATER)**: Add JOLT proving
- Wait for upstream ICME-Lab fix, OR
- Fork and fix dory library, OR
- Try alternative proving systems

**Rationale**: We've completed 90% of the critical work (policies, models, architecture). The build issues are blocking the last 10% (actual proving). Better to ship functional authorization now and add ZK privacy later.

---

## Current Build Status

**What works**:
- ✅ Git operations (no submodule hangs)
- ✅ Cargo index updates (sparse protocol)
- ✅ ~200 crates compile successfully
- ✅ Our code compiles fine

**What fails**:
- ❌ dory library (ark_bn254 version conflicts)
- ❌ Final linking
- ❌ Can't generate JOLT proofs

**Blocking**:
- E2E proving tests
- Performance benchmarks
- Real proof generation

---

## Files Affected

### Working Files ✅
- `policy-examples/onnx/*.py` - All policy transformations (100% complete)
- `policy-examples/onnx/*.onnx` - All ONNX models (exported and validated)
- `jolt-prover/examples/*.rs` - Proving examples (structured, pending build)
- `ZKML_ONLY_ARCHITECTURE.md` - Complete architecture
- `SPENDING_POLICIES_COMPLETE.md` - All spending policies documented

### Blocked Files ❌
- `jolt-prover/target/debug/examples/*` - Can't compile due to dory conflicts
- `jolt-atlas-fork/target/release/*` - Fork build blocked

---

## Next Steps Options

### Option A: Fork dory and fix
```bash
# 1. Fork sxt-dory
# 2. Update Cargo.toml to use ark 0.5.0-alpha.0 consistently
# 3. Add to our Cargo.toml:
[patch."https://github.com/spaceandtimefdn/sxt-dory"]
dory = { git = "https://github.com/USER/sxt-dory-fixed", branch = "fix-ark-versions" }
```

### Option B: Ship without proofs now
```bash
# 1. Build authorization API with ONNX inference
# 2. Skip JOLT proving for MVP
# 3. Add proving in v2
cd ../api
npm init -y
# Build REST API for authorization
```

### Option C: Wait for upstream
```bash
# Check for updates periodically
cd jolt-atlas-fork
git pull origin main
# Test if fixed
```

---

## Conclusion

Your fixes were **excellent** and resolved 90% of the build issues:
- ✅ Git submodule hangs → FIXED
- ✅ Cargo lock contention → FIXED
- ✅ Dependency tree unification → MOSTLY FIXED
- ❌ dory library conflicts → BLOCKED (upstream issue)

The remaining 10% is a deep upstream dependency conflict that requires either:
1. Forking dory library (high effort)
2. Waiting for ICME-Lab fix (uncertain timeline)
3. Shipping without proofs for now (pragmatic)

**Recommended**: Ship Phase 1 without proofs, add proving in Phase 2.
