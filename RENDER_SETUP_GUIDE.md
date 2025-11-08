# Render.com Setup Guide for zkX402

## 🎯 Quick Links

**Your Service:** https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0

**Dashboard:** https://dashboard.render.com

---

## ✅ Required Environment Variables

The `render.yaml` file has been updated with all required variables. However, you can also set them manually in the dashboard:

### Required Variables (Already in render.yaml)

| Variable | Value | Purpose |
|----------|-------|---------|
| `NODE_VERSION` | `18` | Node.js version |
| `PORT` | `10000` | Render assigns this port |
| `NODE_ENV` | `production` | Environment mode |
| **`PAYMENT_WALLET`** | `0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91` | **Your payment wallet** |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `ENABLE_BLOCKCHAIN_MONITOR` | `true` | Enable payment monitoring |
| `REDIS_ENABLED` | `false` | Disable cache (no Redis yet) |

---

## 🔧 Option 1: Automatic (Recommended)

The `render.yaml` file has been updated. Just commit and push:

```bash
git add zkx402-agent-auth/render.yaml
git commit -m "Add payment wallet and environment variables"
git push
```

**Render will automatically:**
- ✅ Detect the updated render.yaml
- ✅ Apply all environment variables
- ✅ Redeploy your service

**No manual configuration needed!**

---

## 🖱️ Option 2: Manual Configuration

If you prefer to set variables manually in the Render dashboard:

### Step 1: Navigate to Environment Settings

1. Go to: https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0
2. Click **"Environment"** tab in the left sidebar

### Step 2: Add Missing Environment Variables

Check if these variables exist. If not, add them:

**Click "Add Environment Variable" for each:**

#### Payment Wallet (CRITICAL - NEW)
```
Key:   PAYMENT_WALLET
Value: 0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
```

#### Logging Level
```
Key:   LOG_LEVEL
Value: info
```

#### Blockchain Monitor
```
Key:   ENABLE_BLOCKCHAIN_MONITOR
Value: true
```

#### Redis Cache (Disabled for now)
```
Key:   REDIS_ENABLED
Value: false
```

### Step 3: Save Changes

1. Click **"Save Changes"** button
2. Service will automatically redeploy
3. Wait 2-5 minutes for deployment to complete

---

## 🚀 Adding Redis Cache (Optional)

If you want to enable proof caching for 600-4800x speedup:

### Step 1: Create Redis Instance

1. Go to Render dashboard: https://dashboard.render.com
2. Click **"New +"** → **"Redis"**
3. Choose plan:
   - **Free:** 25MB (good for ~500 cached proofs)
   - **Starter ($10/mo):** 256MB (~5,000 proofs)
   - **Standard ($25/mo):** 1GB (~20,000 proofs)
4. Click **"Create Redis"**
5. Wait for Redis to provision (~1 minute)

### Step 2: Copy Redis Connection URL

1. Go to your Redis instance
2. Find **"Internal Redis URL"** (looks like: `redis://red-xxxxx:6379`)
3. Copy this URL

### Step 3: Add Redis to Your Service

1. Go back to your web service: https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0
2. Click **"Environment"** tab
3. Update these variables:

```
Key:   REDIS_ENABLED
Value: true
```

```
Key:   REDIS_URL
Value: redis://red-xxxxx:6379  (paste your Redis URL)
```

```
Key:   CACHE_TTL
Value: 86400  (24 hours in seconds)
```

4. Click **"Save Changes"**
5. Service will redeploy with caching enabled

---

## 📋 Complete Environment Variables Checklist

Use this to verify all variables are set correctly:

### Core Settings
- [ ] `NODE_VERSION` = `18`
- [ ] `PORT` = `10000`
- [ ] `NODE_ENV` = `production`

### Payment & Wallet (CRITICAL)
- [ ] `PAYMENT_WALLET` = `0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91`

### Logging & Monitoring
- [ ] `LOG_LEVEL` = `info`
- [ ] `ENABLE_BLOCKCHAIN_MONITOR` = `true`

### Redis Cache (Optional)
- [ ] `REDIS_ENABLED` = `false` (or `true` if Redis added)
- [ ] `REDIS_URL` = `redis://...` (only if Redis enabled)
- [ ] `CACHE_TTL` = `86400` (only if Redis enabled)

### Optional Settings (Usually not needed)
- [ ] `BASE_URL` - Auto-detected from Render
- [ ] `PREFER_NO_DIV` - Leave unset (defaults to 0)

---

## 🔍 Verify Configuration

### Step 1: Check Deployment Status

1. Go to your service: https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0
2. Look for **"Live"** badge (green)
3. Check **"Logs"** tab for any errors

### Step 2: Test Health Endpoint

```bash
# Replace with your actual Render URL
curl https://zkx402-agent-auth.onrender.com/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "service": "zkX402",
  "version": "1.0.0",
  "x402Enabled": true,
  "models": 14,
  "timestamp": "2025-11-08T..."
}
```

### Step 3: Verify Payment Wallet

```bash
# Check x402 discovery endpoint
curl https://zkx402-agent-auth.onrender.com/.well-known/x402 | jq .pricing
```

**Expected output:**
```json
{
  "currency": "USDC",
  "network": "base",
  "chainId": 8453,
  "wallet": "0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91",
  "explorer": "https://basescan.org",
  "tokenAddress": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
}
```

### Step 4: Check Logs

1. Go to **"Logs"** tab in Render dashboard
2. Look for startup message:

```
zkX402 server started {
  port: 10000,
  environment: 'production',
  cacheEnabled: false,
  models: 14
}
```

3. Verify no errors

---

## 🎨 Setting BASE_URL (Optional)

Render automatically provides your app URL. To set it explicitly:

**Find your Render URL:**
1. Go to service dashboard
2. Look at top of page: `https://zkx402-agent-auth.onrender.com`
3. Copy this URL

**Add environment variable:**
```
Key:   BASE_URL
Value: https://zkx402-agent-auth.onrender.com
```

**Note:** Your app will auto-detect the URL from Render, so this is optional.

---

## 🐛 Troubleshooting

### Problem: Deployment Failed

**Check build logs:**
1. Go to **"Logs"** tab
2. Look for errors during build
3. Common issues:
   - Missing dependencies (run `npm install` locally first)
   - Build script errors (check `render-build.sh`)
   - Out of memory (upgrade plan)

**Solution:**
```bash
# Test build locally first
cd zkx402-agent-auth/ui
bash render-build.sh
```

### Problem: Health Check Failing

**Symptoms:**
- Service shows as "Unhealthy"
- Frequent restarts

**Check:**
1. Go to **"Logs"** tab
2. Look for server startup errors
3. Verify `/health` endpoint is accessible

**Solution:**
```bash
# Test health endpoint
curl https://your-app.onrender.com/health
```

### Problem: Payment Wallet Not Working

**Symptoms:**
- x402 endpoint shows wrong wallet address
- Payments not being verified

**Check environment variable:**
1. Go to **"Environment"** tab
2. Verify `PAYMENT_WALLET` exists
3. Value should be: `0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91`

**Fix:**
1. Add/update the variable
2. Click "Save Changes"
3. Wait for redeploy

### Problem: Redis Connection Failed

**Symptoms:**
- Logs show `Redis client error`
- Cache stats show `connected: false`

**Solutions:**

1. **Disable Redis temporarily:**
   ```
   REDIS_ENABLED=false
   ```

2. **Check Redis URL:**
   - Should be: `redis://red-xxxxx:6379`
   - Must be the **Internal URL**, not External

3. **Verify Redis is running:**
   - Go to your Redis instance in dashboard
   - Should show "Available" status

### Problem: Logs Not Showing

**Check:**
1. `LOG_LEVEL` should be `info` (not `error` or higher)
2. `NODE_ENV` should be `production`

**Fix:**
```
LOG_LEVEL=info
NODE_ENV=production
```

---

## 📊 Monitoring Your Service

### View Logs in Real-Time

1. Go to service dashboard
2. Click **"Logs"** tab
3. Logs stream in real-time
4. Use search to filter (e.g., "error", "proof generation")

### View Metrics

1. Click **"Metrics"** tab
2. See:
   - CPU usage
   - Memory usage
   - Request count
   - Response times

### Set Up Alerts (Optional)

1. Click **"Settings"** → **"Alerts"**
2. Configure alerts for:
   - Service downtime
   - High CPU/memory
   - Too many restarts

---

## 🔄 Redeploying After Changes

### Automatic Redeploy (Recommended)

1. Commit and push changes to git
2. Render auto-deploys (if `autoDeploy: true` in render.yaml)
3. Watch deployment in **"Events"** tab

### Manual Redeploy

1. Go to service dashboard
2. Click **"Manual Deploy"** button
3. Select branch (usually `main`)
4. Click **"Deploy"**

---

## 💰 Cost Breakdown

### Current Plan: Standard

**Web Service:**
- **$25/month** for Standard plan
- Includes:
  - 512MB RAM
  - 0.5 CPU
  - Auto-scaling
  - SSL/HTTPS
  - Custom domain

### Optional Add-ons

**Redis Cache:**
- Free: 25MB (~500 proofs) - $0/month
- Starter: 256MB (~5,000 proofs) - $10/month
- Standard: 1GB (~20,000 proofs) - $25/month

**Total Cost Examples:**
- Without Redis: **$25/month**
- With Free Redis: **$25/month**
- With Starter Redis: **$35/month**
- With Standard Redis: **$50/month**

---

## 🎯 Next Steps After Configuration

### 1. Test Your Service

```bash
# Replace with your Render URL
RENDER_URL="https://zkx402-agent-auth.onrender.com"

# Test health
curl $RENDER_URL/health

# Test x402 discovery
curl $RENDER_URL/.well-known/x402

# Test model listing
curl $RENDER_URL/api/policies
```

### 2. Monitor First Payment

1. Send test payment: 0.01 USDC to your wallet
2. Check BaseScan: https://basescan.org/address/0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
3. Watch Render logs for payment verification

### 3. Update Documentation

If you have a custom domain, update:
- `BASE_URL` environment variable
- Documentation with your domain
- x402 discovery endpoint

---

## 📞 Support

### Render Support
- Docs: https://render.com/docs
- Status: https://status.render.com
- Community: https://community.render.com

### zkX402 Issues
- GitHub: https://github.com/hshadab/zkx402/issues
- Check logs first in Render dashboard

---

## ✅ Configuration Complete Checklist

Before going live:

- [ ] `PAYMENT_WALLET` environment variable set
- [ ] Service shows "Live" status
- [ ] Health endpoint returns 200 OK
- [ ] x402 discovery shows correct wallet
- [ ] Test payment sent and verified
- [ ] Logs show no errors
- [ ] (Optional) Redis configured and connected
- [ ] (Optional) Custom domain configured
- [ ] Monitoring/alerts set up

---

**Once all checkboxes are complete, your zkX402 service is ready for production!** 🚀
