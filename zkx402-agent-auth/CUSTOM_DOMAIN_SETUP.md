# Custom Domain Setup Guide for zk-x402.com

## Step 1: Add Custom Domain in Render

1. Go to https://dashboard.render.com/
2. Select your **zkx402** service
3. Navigate to **Settings** → **Custom Domain**
4. Click **Add Custom Domain**
5. Enter: `zk-x402.com`
6. Render will provide DNS configuration instructions

## Step 2: Configure DNS Records

Add these records at your domain registrar (e.g., GoDaddy, Namecheap, Cloudflare):

### For Apex Domain (zk-x402.com)

**Option A: A Record (Recommended)**
```
Type: A
Name: @ (or leave blank)
Value: 216.24.57.1  (Render's IP - verify in Render dashboard)
TTL: 3600 (or Auto)
```

**Option B: CNAME Flattening (if your DNS provider supports it)**
```
Type: CNAME
Name: @ (or leave blank)
Value: zkx402.onrender.com
TTL: 3600 (or Auto)
```

### For WWW Subdomain (Optional)
```
Type: CNAME
Name: www
Value: zkx402.onrender.com
TTL: 3600 (or Auto)
```

### For API Subdomain (Optional - if you want api.zk-x402.com)
```
Type: CNAME
Name: api
Value: zkx402.onrender.com
TTL: 3600 (or Auto)
```

## Step 3: Set Environment Variables in Render

1. In Render dashboard, go to your service
2. Navigate to **Environment** section
3. Add/update these environment variables:

```bash
BASE_URL=https://zk-x402.com
NODE_ENV=production
```

Click **Save Changes** - Render will automatically redeploy.

## Step 4: Wait for DNS Propagation

- DNS changes can take 5 minutes to 48 hours to propagate globally
- Usually takes 15-30 minutes for most regions
- Check status: https://www.whatsmydns.net/#A/zk-x402.com

## Step 5: Enable HTTPS (Automatic)

Render automatically provisions SSL certificates via Let's Encrypt:
- Certificate provisioning starts after DNS is verified
- Takes 1-5 minutes after DNS propagation
- Your site will be accessible via HTTPS automatically

## Verification Checklist

After setup, verify these endpoints work:

- ✅ https://zk-x402.com (React UI)
- ✅ https://zk-x402.com/.well-known/x402 (x402 discovery)
- ✅ https://zk-x402.com/x402/models (Model listing)
- ✅ https://zk-x402.com/health (Health check)

## Troubleshooting

### Domain not resolving
- Wait longer for DNS propagation
- Check DNS records with `dig zk-x402.com`
- Verify records at https://www.whatsmydns.net/

### SSL Certificate Errors
- Render automatically provisions certificates
- May take a few minutes after DNS verification
- Check Render logs for certificate status

### 502 Bad Gateway
- Check if service is running in Render dashboard
- Check deployment logs for errors
- Verify environment variables are set correctly

### API calls failing
- Ensure BASE_URL environment variable is set to https://zk-x402.com
- Check browser console for CORS errors
- Verify all API endpoints use relative URLs

## Current Configuration

- **Service**: zkx402 on Render
- **Old URL**: https://zkx402.onrender.com
- **New URL**: https://zk-x402.com
- **Environment Variable**: `BASE_URL=https://zk-x402.com`

## Notes

- The old Render URL (zkx402.onrender.com) will continue to work
- Render handles traffic for both domains automatically
- No code changes needed - everything uses BASE_URL environment variable
- Payment wallet address remains: `0x1f409E94684804e5158561090Ced8941B47B0CC6`

---

**Last Updated**: 2025-10-29
