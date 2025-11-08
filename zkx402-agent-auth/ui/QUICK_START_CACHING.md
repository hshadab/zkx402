# Quick Start: Caching & Logging

**TL;DR:** Proof caching and structured logging are now active. Cache hits are 600-4800x faster!

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment (Optional)

```bash
# Copy example
cp .env.example .env

# Edit if needed (defaults work fine for development)
```

### 3. Start Server

**With Redis (recommended):**
```bash
# Start Redis first
redis-server

# Start zkX402 server
npm start
```

**Without Redis:**
```bash
REDIS_ENABLED=false npm start
```

---

## 📊 Check Cache Performance

```bash
# Get cache statistics
curl http://localhost:3001/api/cache/stats
```

**Example response:**
```json
{
  "hits": 42,
  "misses": 15,
  "errors": 0,
  "hitRate": "73.68%",
  "enabled": true,
  "connected": true
}
```

---

## 🧪 Test Cache

**1. Generate a proof (cache miss):**
```bash
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": 100,
      "balance": 500
    }
  }'
```
⏱️ Takes 1-6 minutes

**2. Generate same proof again (cache hit):**
```bash
# Same request
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple_threshold",
    "inputs": {
      "amount": 100,
      "balance": 500
    }
  }'
```
⚡ Returns in <100ms with `"cached": true`

---

## 📝 View Logs

```bash
# Watch all logs in real-time
tail -f logs/combined.log | jq .

# Watch only errors
tail -f logs/error.log | jq .

# Search for cache hits
grep "Cache hit" logs/combined.log

# Count proof generations
grep "Proof generation completed" logs/combined.log | wc -l
```

---

## 🔧 Useful Commands

### Clear Cache

```bash
# Clear all cache
curl -X POST http://localhost:3001/api/cache/clear \
  -H "Content-Type: application/json" \
  -d '{}'

# Clear cache for specific model
curl -X POST http://localhost:3001/api/cache/clear \
  -H "Content-Type: application/json" \
  -d '{"modelId": "simple_threshold"}'
```

### Redis Commands

```bash
# Check Redis is running
redis-cli ping

# See all cached proofs
redis-cli KEYS "zkx402:proof:*"

# Count cached proofs
redis-cli DBSIZE

# Clear all cache (alternative)
redis-cli FLUSHDB
```

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_ENABLED` | `true` | Enable/disable caching |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `CACHE_TTL` | `86400` | Cache TTL (24 hours) |
| `NODE_ENV` | `development` | Environment mode |
| `LOG_LEVEL` | `info` | Logging level |

---

## ⚠️ Troubleshooting

### Redis Not Connected

**Symptom:** `"connected": false` in cache stats

**Solutions:**
1. Start Redis: `redis-server`
2. Check Redis is running: `redis-cli ping`
3. Disable caching: `REDIS_ENABLED=false npm start`

### No Cache Hits

**Symptom:** `"hitRate": "0%"`

**Causes:**
- First time running (cache empty)
- Inputs changing every request (expected)
- Cache TTL expired (increase `CACHE_TTL`)
- Redis restarted recently

### Logs Growing Too Large

**Solution:** Winston auto-rotates at 5MB. Max 5 files kept.

---

## 📈 Expected Performance

| Scenario | Time | Cache |
|----------|------|-------|
| First request | 1-8 min | ❌ Miss |
| Same inputs again | <100ms | ✅ Hit |
| Different inputs | 1-8 min | ❌ Miss |

**Cache hit rate expectations:**
- High traffic, repetitive: 60-90%
- Diverse inputs: 10-30%
- Test environments: 80-95%

---

## 📚 Full Documentation

For detailed documentation, see:
- **CACHING_AND_LOGGING.md** - Complete guide
- **IMPLEMENTATION_SUMMARY.md** - Technical overview
- **.env.example** - All environment variables

---

## ✅ Verification Checklist

- [ ] Dependencies installed (`npm install`)
- [ ] Redis running (or `REDIS_ENABLED=false`)
- [ ] Server starts successfully
- [ ] Logs appear in `logs/` directory
- [ ] Cache stats endpoint works
- [ ] First proof request succeeds
- [ ] Second identical request returns cached result

---

**Questions?** See full documentation or open an issue on GitHub.
