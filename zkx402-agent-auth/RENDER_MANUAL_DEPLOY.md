# Manual Render Deployment (Recommended)

Since Render Blueprint is detecting Dockerfiles in subdirectories, the easiest approach is to create the Web Service manually without using the Blueprint feature.

## Step-by-Step Manual Deployment

### 1. Delete the Existing Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Find the `zkx402-agent-auth` service
3. Click on it → **Settings** → **Delete Service**
4. Confirm deletion

### 2. Create New Web Service Manually

1. Click **"New +"** → **"Web Service"** (NOT Blueprint!)
2. Connect your repository: `hshadab/zkx402`
3. Click **"Connect"** next to your repository

### 3. Configure Service Settings

Fill in the following settings:

**Basic Settings:**
- **Name**: `zkx402-agent-auth`
- **Region**: Oregon (US West)
- **Branch**: `main`
- **Root Directory**: `zkx402-agent-auth/ui`

**Build & Deploy:**
- **Runtime**: Node
- **Build Command**:
  ```bash
  bash render-build.sh
  ```
- **Start Command**:
  ```bash
  npm start
  ```

**Plan:**
- Select **Standard** ($25/month) or higher
- ⚠️ DO NOT use Free tier (insufficient RAM for Rust compilation)

**Environment Variables:**
Click "Add Environment Variable" and add:
- `NODE_VERSION` = `18`
- `PORT` = `10000`
- `NODE_ENV` = `production`

**Advanced Settings:**
- **Health Check Path**: `/health`
- **Auto-Deploy**: Yes

### 4. Deploy

1. Click **"Create Web Service"**
2. Render will start building (takes 15-20 minutes first time)

### 5. Monitor Build

Watch the build logs. You should see:
```
========================================
zkX402 Render Build Script
========================================
Installing Rust...
Installing Node.js dependencies...
Building Vite frontend...
Building JOLT Atlas zkML prover...
Compiling proof_json_output binary (this may take 10-15 minutes)...
✅ JOLT Atlas binary built successfully!
```

### 6. Test Deployment

Once deployed, test:
- Health: `https://zkx402-agent-auth.onrender.com/health`
- Discovery: `https://zkx402-agent-auth.onrender.com/.well-known/x402`
- UI: `https://zkx402-agent-auth.onrender.com/`

## Troubleshooting

### Build Still Using Docker?

If you see Docker errors in the build logs:

1. **Check "Root Directory" setting**: Must be `zkx402-agent-auth/ui`
2. **Check "Runtime" setting**: Must be `Node` (not Docker)
3. **Create a `.dockerignore` in root**: Add this file to your repo:
   ```
   **/Dockerfile
   **/.devcontainer
   ```

### Out of Memory During Build

**Error**: `cargo: Killed` or OOM error

**Solution**: Upgrade to Standard plan (4GB RAM minimum)

### Cannot Find jolt-atlas-fork

**Error**: `Cannot find jolt-atlas-fork directory`

**Solution**:
- Verify Root Directory is set to `zkx402-agent-auth/ui`
- The build script uses `cd ../jolt-atlas-fork` which requires the correct root

### Build Takes Too Long

**Issue**: Build timeout after 15 minutes

**Solution**:
- Use Standard plan (no timeout)
- Pre-build the binary and commit it (see main RENDER_DEPLOYMENT.md)

## Why Manual Instead of Blueprint?

The Blueprint (render.yaml) is being overridden by:
1. Existing service configuration from before Blueprint was added
2. Dockerfiles in subdirectories being auto-detected
3. Blueprint settings not fully supported for all use cases

Manual configuration gives you full control and avoids these issues.

## Next Steps After Successful Deploy

1. **Set up monitoring**: Add Sentry or similar
2. **Configure CDN**: Use Cloudflare for static assets
3. **Enable backups**: For any persistent data
4. **Set up CI/CD**: Automatic deploys on push
5. **Add custom domain**: Point your domain to Render

## Support

If you still encounter issues:
1. Share the build logs from Render dashboard
2. Check the [Render Community Forum](https://community.render.com/)
3. Open an issue at [zkX402 GitHub](https://github.com/hshadab/zkx402/issues)
