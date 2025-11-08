# Quick Render Setup - Do This Now!

## 🚀 Choose Your Method

### ✅ EASY WAY (Recommended - 2 minutes)

Just commit the updated `render.yaml` file:

```bash
git add zkx402-agent-auth/render.yaml
git commit -m "Add payment wallet to Render config"
git push
```

**Done!** Render will auto-deploy with all correct environment variables.

---

### 🖱️ MANUAL WAY (If you prefer - 5 minutes)

1. **Go to your service:**
   https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0

2. **Click "Environment" tab**

3. **Add this variable** (if not already there):
   ```
   Name:  PAYMENT_WALLET
   Value: 0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91
   ```

4. **Verify these exist** (should already be set):
   - `NODE_VERSION` = `18`
   - `PORT` = `10000`
   - `NODE_ENV` = `production`

5. **Click "Save Changes"**

6. **Wait 2-5 minutes** for redeploy to complete

---

## ✅ Verify It Worked

### Test 1: Health Check

```bash
curl https://zkx402-agent-auth.onrender.com/health
```

Should return:
```json
{"status": "healthy", "service": "zkX402", ...}
```

### Test 2: Verify Payment Wallet

```bash
curl https://zkx402-agent-auth.onrender.com/.well-known/x402 | grep -A 4 '"pricing"'
```

Should show:
```json
"wallet": "0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91"
```

### Test 3: Check Render Logs

1. Go to: https://dashboard.render.com/web/srv-d413dps9c44c73chf2j0
2. Click "Logs" tab
3. Look for: `zkX402 server started`
4. Should see no errors

---

## ✅ All Environment Variables You Need

| Variable | Value | Required? |
|----------|-------|-----------|
| `NODE_VERSION` | `18` | ✅ Yes |
| `PORT` | `10000` | ✅ Yes (Render default) |
| `NODE_ENV` | `production` | ✅ Yes |
| **`PAYMENT_WALLET`** | `0x2b04D59EdC8Ddfc9b5190902BAf7a2857a103c91` | ✅ **YES - NEW** |
| `LOG_LEVEL` | `info` | ⚙️ Optional (good to have) |
| `ENABLE_BLOCKCHAIN_MONITOR` | `true` | ⚙️ Optional (good to have) |
| `REDIS_ENABLED` | `false` | ⚙️ Optional (disable cache for now) |

**Note:** `BASE_URL` is auto-detected by Render, you don't need to set it!

---

## 📋 Quick Checklist

- [ ] `PAYMENT_WALLET` added to Render environment variables
- [ ] Service redeployed (either automatically or manually)
- [ ] Health check returns 200 OK
- [ ] Wallet address shows in x402 discovery endpoint
- [ ] No errors in Render logs

**Once all checked, you're done!** ✅

---

## 🆘 Problems?

**Service won't start:**
- Check "Logs" tab in Render dashboard
- Look for error messages
- Common fix: Redeploy manually

**Wallet not showing:**
- Verify `PAYMENT_WALLET` is set correctly
- Check for typos in the address
- Redeploy after fixing

**Need more help:**
- See `RENDER_SETUP_GUIDE.md` for detailed instructions
- Check Render logs first

---

**TL;DR:**

1. Commit and push `render.yaml` (easiest)
   OR
2. Add `PAYMENT_WALLET` env var manually on Render

3. Verify it works with `curl` commands above

**That's it!** 🎉
