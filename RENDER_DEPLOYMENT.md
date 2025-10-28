# Deploying zkX402 to Render.com

Complete guide to deploying the zkX402 Agent Authorization system (including JOLT Atlas proof generation) on Render.com.

## Overview

This deployment includes:
- ✅ **JOLT Atlas Rust Prover** - Real zero-knowledge proof generation
- ✅ **Node.js API Server** - REST API for proof generation
- ✅ **React Frontend** - Interactive UI for testing
- ✅ **ONNX Model Support** - All authorization models included
- ✅ **Persistent Storage** - Upload and manage custom models

## Prerequisites

1. **GitHub Account** - Your zkX402 code must be in a GitHub repository
2. **Render Account** - Free at [render.com](https://render.com)
3. **Git Repository** - Code pushed to GitHub (main branch)

## Deployment Methods

### Method 1: Blueprint Deploy (Recommended)

This is the fastest method using Render's infrastructure-as-code.

#### Step 1: Commit Deployment Files

Ensure these files are in your repository:
```bash
git status
# Should show:
# - Dockerfile
# - render.yaml
# - start.sh
# - .dockerignore
```

If not already committed:
```bash
git add Dockerfile render.yaml start.sh .dockerignore zkx402-agent-auth/ui/server.js
git commit -m "Add Render.com deployment configuration"
git push origin main
```

#### Step 2: Deploy to Render

**Option A: Deploy Button (Easiest)**

Click this button to deploy directly:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/hshadab/zkx402)

**Option B: Manual Blueprint Deploy**

1. Log in to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository
4. Select the repository: `hshadab/zkx402`
5. Render will automatically detect `render.yaml`
6. Click **"Apply"**

#### Step 3: Wait for Build

The build process takes approximately **10-15 minutes** and includes:
- ✅ Building Rust JOLT prover (5-7 min)
- ✅ Building React frontend (2-3 min)
- ✅ Creating runtime image (1-2 min)
- ✅ Deploying to Render infrastructure (1-2 min)

**Build Progress**: Monitor in Render Dashboard → Services → zkx402-agent-auth → Logs

#### Step 4: Access Your Deployment

Once deployed, Render provides a URL like:
```
https://zkx402-agent-auth.onrender.com
```

Test the deployment:
```bash
# Health check
curl https://zkx402-agent-auth.onrender.com/api/health

# List models
curl https://zkx402-agent-auth.onrender.com/api/models

# Generate proof (test)
curl -X POST https://zkx402-agent-auth.onrender.com/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_auth",
    "inputs": {
      "amount": "50",
      "balance": "1000",
      "velocity_1h": "20",
      "velocity_24h": "100",
      "vendor_trust": "80"
    }
  }'
```

---

### Method 2: Manual Web Service Deploy

If you prefer manual setup or need customization:

#### Step 1: Create Web Service

1. Log in to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Select: `hshadab/zkx402`

#### Step 2: Configure Service

**Basic Settings**:
- **Name**: `zkx402-agent-auth`
- **Region**: Choose closest to your users (Oregon, Ohio, Virginia, Frankfurt, Singapore)
- **Branch**: `main`
- **Environment**: `Docker`
- **Dockerfile Path**: `./Dockerfile`

**Instance Settings**:
- **Instance Type**: `Standard` or higher (Starter too slow for Rust compilation)
- **Build Command**: (leave empty - Docker handles this)
- **Start Command**: (leave empty - uses CMD from Dockerfile)

**Environment Variables**:
```
NODE_ENV=production
MODELS_DIR=/app/policy-examples/onnx
JOLT_PROVER_DIR=/app/jolt-prover
```

**Advanced Settings**:
- **Health Check Path**: `/api/health`
- **Auto-Deploy**: `Yes` (deploy on git push)

#### Step 3: Optional - Add Persistent Disk

For model uploads:
1. Go to service → **Settings** → **Disks**
2. Click **"Add Disk"**
3. Configure:
   - **Name**: `zkx402-models`
   - **Mount Path**: `/app/policy-examples/onnx`
   - **Size**: `1 GB` (adjust as needed)

#### Step 4: Deploy

Click **"Create Web Service"** and wait for build to complete.

---

## Configuration Options

### Environment Variables

Configure in Render Dashboard → Service → Environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ENV` | `production` | Environment mode |
| `PORT` | `10000` | Server port (auto-set by Render) |
| `MODELS_DIR` | `/app/policy-examples/onnx` | ONNX models directory |
| `JOLT_PROVER_DIR` | `/app/jolt-prover` | JOLT prover binary location |

### Scaling

**Auto-Scaling** (Professional plan):
```yaml
scaling:
  minInstances: 1
  maxInstances: 3
  targetMemoryPercent: 80
  targetCPUPercent: 80
```

Configure in Render Dashboard → Service → Settings → Scaling

**Manual Scaling**:
- Go to Service → Settings → Instance Type
- Upgrade to higher CPU/RAM for faster proof generation

### Custom Domain

1. Go to Service → **Settings** → **Custom Domain**
2. Add your domain (e.g., `zkx402.yourdomain.com`)
3. Update DNS records as instructed
4. SSL automatically provisioned by Render

---

## Performance Considerations

### Build Time Optimization

**First Build**: ~10-15 minutes (Rust compilation is slow)

**Subsequent Builds**: ~5-8 minutes (Docker layer caching)

**Optimization Tips**:
- Use `.dockerignore` to exclude unnecessary files
- Render caches Docker layers between builds
- Upgrade to higher instance type for faster builds

### Runtime Performance

**Proof Generation Times** (on Standard instance):
- Simple Auth: ~1.5-2.0s
- Neural Auth: ~2.5-3.5s

**Recommended Instance Types**:
- **Starter** ($7/month): Development only - very slow for proofs
- **Standard** ($25/month): Good for testing - acceptable performance
- **Pro** ($85/month): Production - faster proofs and better concurrency
- **Pro Plus** ($185/month): High traffic - best performance

### Cost Estimates

**Development**:
- Standard instance: $25/month
- 1GB disk: $0.25/GB/month
- **Total**: ~$26/month

**Production**:
- Pro instance: $85/month
- 5GB disk: $1.25/month
- **Total**: ~$86/month

**Free Tier**: Render offers 750 hours/month free (enough for one service). Use for testing only - insufficient for proof generation.

---

## Monitoring

### Health Checks

Render automatically monitors `/api/health`:
```json
{
  "status": "ok",
  "timestamp": "2025-10-28T09:00:00.000Z",
  "modelsDir": "/app/policy-examples/onnx",
  "modelsAvailable": 5
}
```

If health check fails 3 times, Render restarts the service.

### Logs

**View Logs**:
1. Go to Render Dashboard → Service → **Logs**
2. Real-time streaming logs
3. Filter by level: Info, Error, Warn

**Log Retention**: 7 days (upgrade plan for longer retention)

### Metrics

Available in Render Dashboard → Service → **Metrics**:
- CPU usage
- Memory usage
- Request count
- Response time
- Bandwidth

---

## Troubleshooting

### Build Failures

**Issue**: "Rust compilation out of memory"
```
error: could not compile `zkml-jolt-core`
```

**Solution**: Upgrade to Standard or higher instance type. Starter has insufficient RAM for Rust compilation.

**Issue**: "Cannot find module 'express'"
```
Error: Cannot find module 'express'
```

**Solution**: Check `package.json` is in repository and `npm ci` runs in Dockerfile.

### Runtime Errors

**Issue**: "JOLT prover binary not found"
```
ERROR: JOLT prover binary not found!
```

**Solution**: Check Dockerfile copies prover from builder stage:
```dockerfile
COPY --from=rust-builder /build/jolt-prover/target/release/examples/proof_json_output /app/jolt-prover/target/release/examples/proof_json_output
```

**Issue**: "ONNX models directory not found"
```
ERROR: ONNX models directory not found!
```

**Solution**: Verify policy-examples is copied in Dockerfile:
```dockerfile
COPY --from=rust-builder /build/policy-examples /app/policy-examples
```

**Issue**: Slow proof generation
```
Proof took 30+ seconds
```

**Solution**: Upgrade instance type to Pro or higher. Standard has limited CPU.

### Deployment Issues

**Issue**: "Port already in use"

**Solution**: Render sets PORT environment variable automatically. Don't hardcode port in code.

**Issue**: "502 Bad Gateway"

**Solution**:
1. Check logs for startup errors
2. Verify health check endpoint works
3. Ensure app listens on `process.env.PORT`

---

## Advanced Configuration

### Multiple Environments

Deploy separate services for staging and production:

**Staging**: Deploy from `develop` branch
```yaml
services:
  - type: web
    name: zkx402-staging
    branch: develop
    ...
```

**Production**: Deploy from `main` branch
```yaml
services:
  - type: web
    name: zkx402-production
    branch: main
    ...
```

### CI/CD Integration

Render automatically deploys on git push. To add manual approval:

1. Go to Service → Settings → **Auto-Deploy**
2. Toggle off
3. Use Render API or dashboard to deploy manually

### Database Integration

If adding PostgreSQL for proof history:

```yaml
databases:
  - name: zkx402-db
    databaseName: zkx402
    user: zkx402
    plan: starter

services:
  - type: web
    name: zkx402-agent-auth
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: zkx402-db
          property: connectionString
```

---

## Security Best Practices

1. **Environment Variables**: Store secrets in Render environment variables (not in code)
2. **CORS**: Configure CORS in production:
   ```javascript
   app.use(cors({
     origin: process.env.ALLOWED_ORIGINS?.split(',') || '*'
   }));
   ```
3. **Rate Limiting**: Add rate limiting for production:
   ```javascript
   const rateLimit = require('express-rate-limit');
   const limiter = rateLimit({
     windowMs: 15 * 60 * 1000, // 15 minutes
     max: 100 // limit each IP to 100 requests per windowMs
   });
   app.use('/api/', limiter);
   ```
4. **HTTPS**: Render provides free SSL automatically
5. **DDoS Protection**: Render includes basic DDoS protection

---

## Updating Your Deployment

### Automatic Updates

Render auto-deploys when you push to your configured branch:

```bash
git add .
git commit -m "Update zkX402"
git push origin main
# Render automatically detects push and redeploys
```

### Manual Redeploy

1. Go to Render Dashboard → Service
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Rollback

If a deployment fails:
1. Go to Service → **Events**
2. Find previous successful deploy
3. Click **"Rollback to this version"**

---

## Support and Resources

### Render Documentation
- [Render Docs](https://render.com/docs)
- [Docker Deploys](https://render.com/docs/docker)
- [Blueprint Spec](https://render.com/docs/blueprint-spec)

### zkX402 Resources
- [GitHub Repository](https://github.com/hshadab/zkx402)
- [API Reference](./API_REFERENCE.md)
- [Quickstart Guide](./QUICKSTART.md)

### Getting Help
- Render Support: support@render.com
- Render Community: [community.render.com](https://community.render.com)
- zkX402 Issues: [GitHub Issues](https://github.com/hshadab/zkx402/issues)

---

## Next Steps

After successful deployment:

1. ✅ **Test the API**: Use the provided curl commands
2. ✅ **Configure custom domain**: Add your domain in Render settings
3. ✅ **Set up monitoring**: Configure alerts in Render dashboard
4. ✅ **Integrate with x402**: Update x402 payment protocol to use your deployed API
5. ✅ **Scale as needed**: Monitor usage and upgrade instance type

---

**Status**: Ready to deploy! 🚀

**Estimated Time**: 15 minutes (after git push)

**Estimated Cost**: $25-85/month (depending on instance type)
