# Deploying zkX402 to Render.com

This guide explains how to deploy the zkX402 Agent Authorization system to Render.com, including building the JOLT Atlas zkML prover binary.

## Prerequisites

- Render.com account
- GitHub repository connected to Render
- **Important**: Use a **Standard** or higher plan (not Free tier) because:
  - JOLT Atlas compilation takes 10-15 minutes (Free tier has 15min build timeout)
  - Requires at least 4GB RAM for Rust compilation
  - zkML proof generation is CPU-intensive

## Deployment Steps

### 1. Connect Your Repository

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +" → "Blueprint"**
3. Connect your GitHub repository: `hshadab/zkx402`
4. Select branch: `main`

### 2. Configure the Service

The `render.yaml` Blueprint file is already configured with:

```yaml
services:
  - type: web
    name: zkx402-agent-auth
    env: node
    region: oregon
    plan: standard  # Required for Rust compilation
    buildCommand: cd ui && bash render-build.sh
    startCommand: cd ui && npm start
    envVars:
      - key: NODE_VERSION
        value: 18
      - key: PORT
        value: 10000
      - key: NODE_ENV
        value: production
```

### 3. Build Process

The `render-build.sh` script will:

1. **Install Rust** (if not present)
   - Uses rustup installer
   - Adds cargo to PATH

2. **Install Node dependencies**
   ```bash
   npm install
   ```

3. **Build Vite frontend**
   ```bash
   npm run build
   ```

4. **Compile JOLT Atlas zkML prover binary**
   ```bash
   cd ../jolt-atlas-fork/zkml-jolt-core
   cargo build --release --example proof_json_output
   ```
   - This takes **10-15 minutes** on Render Standard plan
   - Creates binary at: `jolt-atlas-fork/target/release/examples/proof_json_output`

### 4. Deploy

1. Click **"Apply"** in Render Blueprint view
2. Render will:
   - Clone your repository
   - Run the build script
   - Start the service on port 10000

### 5. Verify Deployment

Once deployed, test these endpoints:

- **Health Check**: `https://your-app.onrender.com/health`
- **x402 Discovery**: `https://your-app.onrender.com/.well-known/x402`
- **Models List**: `https://your-app.onrender.com/x402/models`
- **UI**: `https://your-app.onrender.com/`

## Important Notes

### Build Time
- **First build**: 15-20 minutes (includes Rust installation + JOLT compilation)
- **Subsequent builds**: 10-15 minutes (Rust cached, but JOLT recompiles)

### Memory Requirements
- **Build**: 4-8 GB RAM (Rust + JOLT compilation)
- **Runtime**: 2-4 GB RAM (Node.js + proof generation)

### Proof Generation Performance
- Render Standard plan has 4 vCPUs
- Proof generation: 6-10 seconds (proving) + 6 minutes (verification)
- Consider upgrading to **Pro** plan for better performance

### Storage
- JOLT binary: ~137MB
- Total deployment: ~500MB

## Troubleshooting

### Build Timeout
**Error**: Build timeout after 15 minutes

**Solution**: Upgrade to Standard plan (Free tier has 15min limit)

### Out of Memory During Build
**Error**: `cargo: Killed` or OOM error

**Solution**:
- Upgrade to Standard plan (Free tier has 512MB RAM)
- Standard plan has 4GB RAM which is sufficient

### Binary Not Found
**Error**: `Model file not found` or proof generation fails

**Solution**: Check that `render-build.sh` completed successfully:
```bash
# In build logs, look for:
✅ JOLT Atlas binary built successfully!
```

### Slow Proof Generation
**Issue**: Proofs taking >10 minutes

**Solution**:
- This is normal on Standard plan (4 vCPUs)
- Upgrade to Pro plan (8 vCPUs) for faster proofs
- Consider implementing proof caching

## Cost Estimate

### Render Standard Plan
- **Cost**: $25/month
- **Resources**: 4GB RAM, 4 vCPUs
- **Build**: Sufficient for compilation
- **Runtime**: Good for demo/development

### Render Pro Plan
- **Cost**: $85/month
- **Resources**: 8GB RAM, 8 vCPUs
- **Build**: Fast compilation
- **Runtime**: Better proof generation performance

## Alternative: Pre-build Binary

To reduce build time, you can pre-build the binary and commit it:

### 1. Build Locally
```bash
cd jolt-atlas-fork/zkml-jolt-core
cargo build --release --example proof_json_output
```

### 2. Commit Binary
```bash
git add jolt-atlas-fork/target/release/examples/proof_json_output
git commit -m "Add pre-built JOLT Atlas binary"
git push
```

### 3. Update render-build.sh
Skip the cargo build step if binary exists:

```bash
if [ ! -f "../jolt-atlas-fork/target/release/examples/proof_json_output" ]; then
    echo "Building JOLT Atlas binary..."
    cd ../jolt-atlas-fork/zkml-jolt-core
    cargo build --release --example proof_json_output
else
    echo "Using pre-built JOLT Atlas binary"
fi
```

**Pros**:
- Much faster builds (2-3 minutes instead of 15)
- Can use Free tier for deployment

**Cons**:
- Binary is 137MB (increases repo size)
- Need to rebuild locally for updates
- Binary must be built for Linux x64

## Production Recommendations

For production deployment:

1. **Use Standard or Pro plan** for better performance
2. **Enable automatic deploys** from `main` branch
3. **Set up monitoring**:
   - Health check endpoint: `/health`
   - Proof generation metrics
   - Error tracking (Sentry, etc.)
4. **Implement proof caching** to reduce computation
5. **Consider CDN** for frontend assets (Cloudflare, etc.)
6. **Database** for proof/transaction history (optional)

## Support

- [Render Documentation](https://render.com/docs)
- [JOLT Atlas Repository](https://github.com/your-jolt-atlas-fork)
- [zkX402 Issues](https://github.com/hshadab/zkx402/issues)
