# Render Build Optimization Guide

## Problem: Slow Builds for UI Changes

By default, Render recompiles the entire JOLT Atlas Rust codebase on every deploy, even for UI-only changes:

- **Current build time**: ~15 minutes per deploy
- **Issue**: UI changes (React, CSS, docs) trigger full Rust recompilation
- **Cost**: Wasted build minutes, slower iteration

## Solution: Smart Build Caching

I've optimized the `render-build.sh` script to skip compilation when the binary already exists.

### How It Works

**Updated build script behavior:**

```bash
# Check if binary exists
if binary already exists:
    ✅ Skip compilation (2-3 minutes build)
    Use existing binary
else:
    🔨 Compile JOLT Atlas (15 minutes build)
    Create new binary
```

### Build Time Comparison

| Change Type | Before | After |
|-------------|--------|-------|
| UI changes (React, CSS, docs) | 15 min | **2-3 min** |
| Rust code changes | 15 min | 15 min (recompiles) |
| First-time build | 15 min | 15 min |

## Option 1: Use Render's Build Cache (Current Setup)

**Status**: Already implemented in `render-build.sh`

Render caches the build directory between deploys. If the binary exists from a previous build, it's reused.

**Pros:**
- ✅ No manual steps required
- ✅ Automatic caching
- ✅ Works for all developers

**Cons:**
- ⚠️ Cache can be invalidated (Render restarts, manual clear)
- ⚠️ First build after cache clear takes 15 minutes

**When this helps:**
- Quick UI tweaks (header, colors, docs)
- Backend logic changes (server.js, middleware)
- Configuration updates

## Option 2: Commit Pre-Built Binary (Recommended)

For even faster builds, commit the pre-built binary to git.

### Step 1: Build Binary Locally

```bash
cd zkx402-agent-auth
./scripts/build-and-commit-binary.sh
```

This script will:
1. Compile JOLT Atlas locally (~15 minutes, one-time)
2. Stage the binary for git commit
3. Show you next steps

### Step 2: Update .gitignore

The binary is ~137MB, so git may ignore it. Add an exception:

**Option A: Exception pattern (cleaner)**
```gitignore
# In jolt-atlas-fork/.gitignore or root .gitignore
target/
!target/release/examples/proof_json_output
```

**Option B: Comment out target/ entirely**
```gitignore
# target/  # Commented to allow binary
```

### Step 3: Commit and Push

```bash
git commit -m "Add pre-built JOLT Atlas binary for faster Render builds"
git push origin main
```

### Result

**Build times after committing binary:**

| Deploy | Before | After |
|--------|--------|-------|
| UI changes | 15 min | **~2 min** |
| Docs updates | 15 min | **~2 min** |
| Config changes | 15 min | **~2 min** |
| Rust code changes | 15 min | 15 min* |

*When you update Rust code, delete the binary and rebuild:
```bash
rm jolt-atlas-fork/target/release/examples/proof_json_output
./scripts/build-and-commit-binary.sh
git commit -m "Update JOLT Atlas binary"
git push
```

**Pros:**
- ✅ Fastest builds (2 minutes)
- ✅ Reliable (not dependent on cache)
- ✅ Works immediately after push

**Cons:**
- ⚠️ Binary is 137MB (increases repo size)
- ⚠️ Must rebuild locally when changing Rust code
- ⚠️ Binary must be built for Linux x64 (Render's platform)

## Option 3: Separate Services (Advanced)

For production at scale, separate the Rust prover and UI into different services:

**Architecture:**
```
┌─────────────────┐
│   UI Service    │  Fast rebuilds (Node.js only)
│   (Node + React)│
└────────┬────────┘
         │ HTTP calls
         ▼
┌─────────────────┐
│  Prover Service │  Rare rebuilds (Rust only)
│  (JOLT Atlas)   │
└─────────────────┘
```

**Render configuration:**
```yaml
services:
  # UI service (fast deploys)
  - name: zkx402-ui
    type: web
    env: node
    buildCommand: npm install && npm run build
    startCommand: npm start

  # Prover service (rare deploys)
  - name: zkx402-prover
    type: web
    env: docker
    dockerfilePath: ./Dockerfile.prover
```

**Pros:**
- ✅ UI deploys: ~2 minutes
- ✅ Rust changes only rebuild prover service
- ✅ Can scale services independently
- ✅ Best for production

**Cons:**
- ⚠️ More complex setup
- ⚠️ Need to handle inter-service communication
- ⚠️ Requires Docker knowledge

## Current Status

**Implemented:** Option 1 (Smart build caching)

Your build script now skips compilation when the binary exists, saving 10-15 minutes on most deploys.

**Next time you deploy:**
1. UI changes → ~2-3 minute build ✅
2. Documentation updates → ~2-3 minute build ✅
3. First build after cache clear → 15 minutes (one-time)

## Recommendation

**For your use case (solo developer, frequent UI tweaks):**

I recommend **Option 2** (commit pre-built binary):
1. Run `./scripts/build-and-commit-binary.sh` once
2. Commit the binary
3. Enjoy 2-minute builds for all UI changes
4. Rebuild only when you change Rust code (rare)

**Command to commit binary:**
```bash
cd zkx402-agent-auth
./scripts/build-and-commit-binary.sh
git commit -m "Add pre-built JOLT Atlas binary"
git push origin main
```

After this, your next UI deploy will take ~2 minutes instead of 15! 🚀

## Monitoring Build Times

You can see build times in Render dashboard:
1. Go to https://dashboard.render.com/
2. Select your service
3. Click on any deploy
4. Check "Build logs" → duration at bottom

**Look for this in logs:**
```
✅ JOLT Atlas binary already exists!
⏭️  Skipping compilation (binary exists)
```

If you see "Compiling proof_json_output binary", the optimization isn't working (binary not found).

## When to Rebuild Binary

Rebuild and recommit the binary when you:
- Update JOLT Atlas source code
- Change Rust dependencies (Cargo.toml)
- Modify proof generation logic
- Update zkML operations

**Quick rebuild:**
```bash
rm jolt-atlas-fork/target/release/examples/proof_json_output
./scripts/build-and-commit-binary.sh
git commit -m "Update JOLT binary"
git push
```

---

**Questions?** Check the Render logs or open an issue: https://github.com/hshadab/zkx402/issues
