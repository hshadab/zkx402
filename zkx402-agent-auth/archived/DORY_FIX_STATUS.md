# Dory Fix Status

**Date**: 2025-10-27

---

## Progress Update

### ✅ Problem 1: SOLVED - Dory Arkworks Version Conflicts

**Original issue**: Two versions of ark_bn254 in dependency tree
- `ark-bn254 v0.5.0` (patched from crates.io)
- `ark-bn254 v0.5.0-alpha.0` (from jolt-optimizations)

**Solution implemented**:
1. Forked `https://github.com/spaceandtimefdn/sxt-dory`
2. Created branch `fix-ark-version-conflicts`
3. Changed all ark-* dependencies to use git sources directly:
```toml
ark-bn254 = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-ec = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout", ... }
ark-ff = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout", ... }
ark-serialize = { git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }
ark-std = "0.5.0"  # Not in a16z fork, use crates.io
```
4. Removed [patch.crates-io] section (no longer needed)
5. Patched jolt-prover to use our fixed dory:
```toml
[patch."https://github.com/spaceandtimefdn/sxt-dory"]
dory = { path = "../dory-fork" }
```

**Result**: ✅ Dory compiles successfully! No more type mismatches.

---

### ❌ Problem 2: NEW - ICME-Lab jolt-core API Mismatch

**New issue**: `jolt-core` from ICME-Lab/zkml-jolt uses old arkworks MSM API

**Error**:
```rust
error[E0061]: this function takes 3 arguments but 2 arguments were supplied
  --> jolt-core/src/fast_msm/mod.rs:37:25
   |
37 |     msm_binary::<Self>(bases, &bool_scalars)
   |     ^^^^^^^^^^^^^^^^^^---------------------- argument #3 of type `bool` is missing

note: function defined here
  --> arkworks-algebra/ec/src/scalar_mul/variable_base/mod.rs:451:8
   |
451| pub fn msm_binary<V: VariableBaseMSM>(
   |        ^^^^^^^^^^
```

**Root cause**: The a16z arkworks fork added a `serial: bool` parameter to MSM functions, but ICME-Lab/zkml-jolt hasn't updated their jolt-core to match.

**Affected functions**:
- `msm_binary()` - expects 3 args, jolt-core passes 2
- `msm_u8()` - expects 3 args, jolt-core passes 2
- `msm_u16()` - expects 3 args, jolt-core passes 2
- `msm_u32()` - expects 3 args, jolt-core passes 2
- `msm_u64()` - expects 3 args, jolt-core passes 2

**File locations**: `/jolt-core/src/fast_msm/mod.rs`

---

## Solutions for Problem 2

### Option 1: Fork and Fix ICME-Lab/zkml-jolt ⚡ (Medium effort, recommended)

Fork `https://github.com/ICME-Lab/zkml-jolt` and update MSM calls:

```rust
// Before
msm_binary::<Self>(bases, &bool_scalars)

// After
msm_binary::<Self>(bases, &bool_scalars, false)  // false = parallel (default)
```

Apply to all 5+ MSM call sites in fast_msm/mod.rs.

**Pros**:
- Complete control
- Small targeted fix
- Maintains MAX_TENSOR_SIZE=1024

**Cons**:
- Another fork to maintain
- Need to sync with upstream

**Implementation**:
```bash
cd ..
git clone https://github.com/ICME-Lab/zkml-jolt zkml-jolt-fork
cd zkml-jolt-fork
# Edit jolt-core/src/fast_msm/mod.rs
# Add false parameter to all msm_* calls
# Commit and push
```

Then update jolt-prover Cargo.toml:
```toml
[dependencies]
jolt-core = { git = "https://github.com/USER/zkml-jolt-fork", branch = "fix-msm-api", ... }
```

---

### Option 2: Use Older Arkworks (Compatible with ICME-Lab) ⚠️ (Quick but risky)

Try finding an older commit of a16z/arkworks-algebra that still has the 2-arg MSM API.

**Pros**:
- No code changes needed
- Quick to test

**Cons**:
- May not be compatible with latest dory
- May lose important fixes/optimizations
- Hard to find the right commit

---

### Option 3: Wait for ICME-Lab to Update ⏰ (Zero effort, uncertain timeline)

The ICME-Lab team may already be working on this or may update if we report it.

**Pros**:
- Zero effort
- Proper long-term solution

**Cons**:
- Unknown timeline
- Project may be inactive
- Blocks our progress

---

### Option 4: Ship Without Proofs (Already discussed) 🚀

As we already discussed, we can ship authorization without ZK proofs initially.

**Pros**:
- Unblocks shipping immediately
- All models trained and validated
- Can add proofs in v2

**Cons**:
- No privacy (no ZK)
- Not the full vision

---

## Recommendation

**Immediate next step**: Fork and fix ICME-Lab/zkml-jolt (Option 1)

**Why**:
1. We've already successfully fixed dory (validated!)
2. The MSM fix is straightforward (add `false` parameter)
3. Small targeted change (~5-10 lines)
4. Maintains our MAX_TENSOR_SIZE=1024 optimization
5. We're 95% there - this is the last blocker!

**Alternative**: If MSM fix reveals more issues, pivot to Option 4 (ship without proofs)

---

## Current Status

### What Works ✅
- ✅ MAX_TENSOR_SIZE=1024 fork created
- ✅ All 6 policies transformed (98-100% accuracy)
- ✅ All ONNX models exported and validated
- ✅ Dory library fixed (arkworks conflicts resolved)
- ✅ Dory compiles successfully
- ✅ Git/cargo issues resolved

### What's Blocked ❌
- ❌ jolt-core compilation (MSM API mismatch)
- ❌ Final linking
- ❌ E2E proving tests
- ❌ Performance benchmarks

### Files Modified
- ✅ `dory-fork/Cargo.toml` - Fixed arkworks dependencies
- ✅ `jolt-prover/Cargo.toml` - Added dory patch
- ⏳ Need: `zkml-jolt-fork/jolt-core/src/fast_msm/mod.rs` - Fix MSM calls

---

## Next Action

```bash
# 1. Fork ICME-Lab/zkml-jolt
cd /home/hshadab/zkx402/zkx402-agent-auth
git clone https://github.com/ICME-Lab/zkml-jolt zkml-jolt-fork
cd zkml-jolt-fork
git checkout zkml-jolt

# 2. Fix MSM calls in jolt-core/src/fast_msm/mod.rs
# Add false parameter to all msm_* function calls

# 3. Update jolt-prover/Cargo.toml to use our fork
[dependencies]
jolt-core = { path = "../zkml-jolt-fork/jolt-core", ... }

# 4. Test build
cd ../jolt-prover
cargo build --example simple_velocity_e2e
```

---

## Timeline Estimate

- **Option 1 (Fork jolt-core)**: 30-60 minutes
- **Option 4 (Ship without proofs)**: Can start immediately

**Let's try Option 1 first!** We're so close - one more fix and we'll have working JOLT proofs! 🚀
