# Caching and Logging Features

This document describes the structured logging and proof caching features added to zkX402.

## Table of Contents

- [Structured Logging with Winston](#structured-logging-with-winston)
- [Proof Result Caching with Redis](#proof-result-caching-with-redis)
- [Configuration](#configuration)
- [Monitoring and Metrics](#monitoring-and-metrics)
- [Deployment Considerations](#deployment-considerations)

---

## Structured Logging with Winston

### Overview

The application now uses [Winston](https://github.com/winstonjs/winston) for structured, production-grade logging instead of `console.log`/`console.error`.

### Features

- **Structured JSON logging** in production
- **Human-readable colored logs** in development
- **Multiple log levels**: error, warn, info, http, debug
- **Log rotation** with automatic file management
- **Separate error logs** for quick troubleshooting
- **Exception and rejection handlers**
- **Request context logging** (IP, request ID, model ID)

### Log Files

Logs are stored in `zkx402-agent-auth/ui/logs/`:

```
logs/
├── combined.log      # All log levels
├── error.log         # Only errors (warn and error)
├── exceptions.log    # Uncaught exceptions
└── rejections.log    # Unhandled promise rejections
```

**File Rotation:**
- Max file size: 5MB
- Max files kept: 5
- Automatic compression and rotation

### Log Levels

| Level | When to Use | Example |
|-------|------------|---------|
| `error` | System errors, exceptions | Payment verification failed |
| `warn` | Warnings, missing data | Cache unavailable, invalid inputs |
| `info` | Important events | Proof generated, server started |
| `http` | HTTP requests/responses | API endpoint calls |
| `debug` | Detailed debugging info | Cache keys, intermediate values |

### Log Format

**Production (JSON):**
```json
{
  "timestamp": "2025-11-08 14:23:45:123",
  "level": "info",
  "message": "Proof generation completed",
  "metadata": {
    "modelId": "simple_threshold",
    "requestId": "req_1699454625123_abc123",
    "proofTime": "125000ms",
    "approved": true
  }
}
```

**Development (Human-readable):**
```
2025-11-08 14:23:45 [info]: Proof generation completed {"modelId":"simple_threshold","requestId":"req_1699454625123_abc123","proofTime":"125000ms","approved":true}
```

### Environment Variables

```bash
# Set log level (default: info in production, debug in development)
LOG_LEVEL=info

# Set environment
NODE_ENV=production
```

### Key Logging Points

1. **Server Startup**
   - Redis connection status
   - Number of models loaded
   - Port and environment

2. **Proof Generation**
   - Request received (model ID, IP)
   - Cache hit/miss
   - Proof generation started
   - Proof generation completed (timing, approval)
   - Errors with full context

3. **x402 Payments**
   - Payment verification started
   - Payment verification result
   - zkML proof verification result

4. **Cache Operations**
   - Cache hits/misses
   - Keys generated
   - Cache errors

### Production Security

- **Stack traces hidden** in production (only shown in development)
- **Sensitive data excluded** from logs (payment details sanitized)
- **Request IDs** for tracking without exposing user data

---

## Proof Result Caching with Redis

### Overview

Proof results are cached in Redis to avoid regenerating identical proofs (1-8 minute operation → instant response).

### How It Works

1. **Request arrives** with model ID + inputs
2. **Hash generated** from `modelId:inputs` (SHA-256)
3. **Cache lookup** in Redis with key `zkx402:proof:{hash}`
4. **Cache hit?** → Return cached proof instantly
5. **Cache miss?** → Generate proof, store in cache, return result

### Cache Key Generation

```javascript
// Example: simple_threshold with amount=100, balance=500
modelId = "simple_threshold"
inputs = { amount: 100, balance: 500 }

// Deterministic string
inputsStr = "100,500"

// SHA-256 hash
hash = sha256("simple_threshold:100,500")
     = "a3f7b2c1..."

// Redis key
key = "zkx402:proof:a3f7b2c1..."
```

### Cached Data Structure

```json
{
  "approved": true,
  "output": 1,
  "verification": { ... },
  "proofSize": "1.2 MB",
  "verificationTime": "4.2 seconds",
  "operations": 6,
  "zkmlProof": { ... },
  "modelId": "simple_threshold",
  "modelName": "Simple Threshold",
  "cachedAt": "2025-11-08T14:23:45.123Z"
}
```

### Cache Configuration

```bash
# Enable/disable caching (default: true)
REDIS_ENABLED=true

# Redis connection URL
REDIS_URL=redis://localhost:6379

# Time-to-live for cache entries (seconds, default: 86400 = 24 hours)
CACHE_TTL=86400
```

### Cache Features

- **Automatic reconnection** with exponential backoff
- **Graceful degradation** (if Redis is down, proofs still generate)
- **Hit/miss metrics** tracked
- **Error logging** without breaking proof generation
- **TTL-based expiration** (default: 24 hours)

### API Endpoints

#### Get Cache Statistics

```bash
GET /api/cache/stats
```

**Response:**
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

#### Clear Cache

```bash
# Clear all cache
POST /api/cache/clear
Content-Type: application/json
{}

# Clear cache for specific model
POST /api/cache/clear
Content-Type: application/json
{
  "modelId": "simple_threshold"
}
```

**Response:**
```json
{
  "success": true,
  "deletedCount": 42,
  "modelId": "simple_threshold"
}
```

### Cache Hit Response

When a proof is returned from cache, the response includes:

```json
{
  "approved": true,
  "output": 1,
  "cached": true,  // ← Indicates cache hit
  "cachedAt": "2025-11-08T14:23:45.123Z",
  "request_id": "req_...",
  ...
}
```

### Performance Impact

| Scenario | Without Cache | With Cache (Hit) | Improvement |
|----------|--------------|------------------|-------------|
| Simple threshold | 1-6.5 minutes | <100ms | **600x - 3900x faster** |
| Advanced neural | 5-8 minutes | <100ms | **3000x - 4800x faster** |

**Cost Savings:**
- Repeated authorization checks (same inputs) = FREE
- No proof regeneration cost
- Reduced server load

### When Cache Invalidation Occurs

1. **TTL expiration** (default: 24 hours)
2. **Manual clearing** via `/api/cache/clear`
3. **Model update** (clear specific model cache)
4. **Redis restart** (in-memory cache cleared)

### Cache Hit Rate Expectations

- **High traffic, repetitive scenarios**: 60-90% hit rate
- **Diverse inputs**: 10-30% hit rate
- **Test environments**: 80-95% hit rate

---

## Configuration

### Environment Variables (.env)

```bash
# Server
PORT=3001
BASE_URL=http://localhost:3001
NODE_ENV=production

# Redis
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379
CACHE_TTL=86400

# Logging
LOG_LEVEL=info
```

### Redis Installation

**Local Development:**

```bash
# macOS (Homebrew)
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:7-alpine
```

**Production (Render.com):**

1. Go to Render Dashboard
2. Create New → Redis
3. Select plan (Free tier available: 25MB, good for ~1000 cached proofs)
4. Copy connection URL
5. Add to environment variables: `REDIS_URL=redis://:password@host:port`

**Production (Railway):**

```bash
railway add redis
railway variables  # Copy REDIS_URL
```

### Running Without Redis

If Redis is not available, the system gracefully degrades:

1. Cache is disabled automatically
2. All proofs are generated fresh
3. Warnings logged (not errors)
4. No impact on core functionality

```bash
# Disable caching explicitly
REDIS_ENABLED=false
```

---

## Monitoring and Metrics

### Health Check

```bash
GET /health
```

Includes cache status:
```json
{
  "status": "healthy",
  "service": "zkX402",
  "version": "1.0.0",
  "x402Enabled": true,
  "models": 14,
  "timestamp": "2025-11-08T14:23:45.123Z"
}
```

### Analytics Integration

Cache metrics are automatically logged to analytics:

```javascript
analyticsManager.logRequest({
  endpoint: '/api/generate-proof',
  method: 'POST',
  modelId: 'simple_threshold',
  success: true,
  responseTime: 0,  // <-- 0ms for cache hits
  cached: true,      // <-- Cache hit indicator
  userAgent: '...',
  ip: '...'
});
```

### Prometheus-Ready (Future Enhancement)

The logging structure is compatible with Prometheus exporters:

```javascript
// Future metrics export
cache_hits_total{model="simple_threshold"} 42
cache_misses_total{model="simple_threshold"} 15
cache_hit_rate{model="simple_threshold"} 0.7368
proof_generation_time_seconds{model="simple_threshold"} 125.0
```

---

## Deployment Considerations

### Production Checklist

- [ ] Set `NODE_ENV=production`
- [ ] Configure `REDIS_URL` for production Redis instance
- [ ] Set appropriate `CACHE_TTL` (default: 24h is good)
- [ ] Set `LOG_LEVEL=info` (avoid debug in production)
- [ ] Ensure logs directory is writable (or configure external logging)
- [ ] Monitor Redis memory usage (use Redis `maxmemory` policy)
- [ ] Set up log rotation/aggregation (e.g., Datadog, Loggly, CloudWatch)

### Redis Memory Management

**Estimate Cache Size:**
- Average proof: ~50KB
- 1000 proofs: ~50MB
- 10,000 proofs: ~500MB

**Redis Eviction Policy:**

Add to `redis.conf` or set via CLI:

```bash
# Set max memory (e.g., 100MB)
maxmemory 100mb

# Eviction policy (delete oldest keys when full)
maxmemory-policy allkeys-lru
```

**Render.com Free Tier:**
- 25MB limit
- ~500 cached proofs
- Good for testing/low-traffic production

**Render.com Paid Tier:**
- $10/month: 256MB (~5,000 proofs)
- $25/month: 1GB (~20,000 proofs)

### Log Aggregation

**Production Recommendations:**

1. **Datadog** (Application Performance Monitoring)
   ```bash
   npm install winston-datadog
   ```

2. **Loggly** (Log Management)
   ```bash
   npm install winston-loggly-bulk
   ```

3. **CloudWatch** (AWS)
   ```bash
   npm install winston-cloudwatch
   ```

4. **File-based** (keep for debugging)
   - Mount logs volume in Docker
   - Set up log rotation with `logrotate`

### Graceful Shutdown

The server now handles graceful shutdown:

```bash
# SIGTERM (Docker, Kubernetes, Render)
kill -SIGTERM <pid>

# SIGINT (Ctrl+C)
^C
```

**Shutdown sequence:**
1. Log shutdown signal received
2. Close Redis connection gracefully
3. Exit with code 0

### Horizontal Scaling

The caching layer is **shared across instances** if using a centralized Redis:

```
┌─────────────┐
│  Instance 1 │─┐
└─────────────┘ │
                ├──→ Redis (shared cache)
┌─────────────┐ │
│  Instance 2 │─┘
└─────────────┘
```

**Benefits:**
- Cache hits across all instances
- No duplicate proof generation
- Efficient resource utilization

---

## Troubleshooting

### Redis Connection Issues

**Symptom:** `Redis client error` in logs

**Solutions:**
1. Check Redis is running: `redis-cli ping` (should return `PONG`)
2. Verify `REDIS_URL` is correct
3. Check firewall/network rules
4. Disable caching temporarily: `REDIS_ENABLED=false`

### High Cache Miss Rate

**Symptom:** Cache hit rate < 20%

**Possible causes:**
1. Inputs are highly variable (expected)
2. Cache TTL too short (increase `CACHE_TTL`)
3. Cache being cleared frequently
4. Redis restarting (check Redis logs)

**Investigation:**
```bash
# Check cache stats
curl http://localhost:3001/api/cache/stats

# Check Redis memory
redis-cli INFO memory

# Check Redis key count
redis-cli DBSIZE
```

### Log File Growing Too Large

**Symptom:** `logs/combined.log` exceeds disk space

**Solutions:**
1. Winston auto-rotates at 5MB (default)
2. Reduce `LOG_LEVEL` in production (use `info`, not `debug`)
3. Set up external log aggregation
4. Manually rotate: `logrotate /path/to/zkx402/ui/logs`

### Stack Traces in Production

**Symptom:** Sensitive error stacks exposed to users

**Solution:** Ensure `NODE_ENV=production` is set (stack traces are automatically hidden)

---

## Performance Testing

### Test Cache Hit Rate

```bash
# Generate proof (cache miss)
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{"model":"simple_threshold","inputs":{"amount":100,"balance":500}}'

# Wait 1-8 minutes...

# Generate same proof (cache hit)
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{"model":"simple_threshold","inputs":{"amount":100,"balance":500}}'

# Should return instantly with "cached": true
```

### Monitor Logs

```bash
# Watch logs in real-time (development)
tail -f logs/combined.log | jq .

# Watch only errors
tail -f logs/error.log | jq .

# Search for cache hits
grep "Cache hit" logs/combined.log | jq .
```

### Check Redis Keys

```bash
# Connect to Redis
redis-cli

# List all zkX402 keys
KEYS zkx402:proof:*

# Get specific proof
GET zkx402:proof:a3f7b2c1...

# Check TTL
TTL zkx402:proof:a3f7b2c1...

# Delete all cache
FLUSHDB
```

---

## Summary

### Key Improvements

1. **Structured Logging (Winston)**
   - Production-grade logging with rotation
   - Contextual metadata for debugging
   - Separate error logs
   - Stack traces hidden in production

2. **Proof Caching (Redis)**
   - 600x - 4800x faster for repeated proofs
   - Automatic cache management
   - Graceful degradation if Redis unavailable
   - Hit/miss metrics

3. **New API Endpoints**
   - `GET /api/cache/stats` - Cache performance
   - `POST /api/cache/clear` - Cache management

4. **Graceful Shutdown**
   - Proper Redis connection cleanup
   - SIGTERM/SIGINT handlers

### Migration Path

**Existing deployments:**

1. Install dependencies: `npm install winston redis`
2. Add environment variables (see `.env.example`)
3. Optional: Set up Redis (or use `REDIS_ENABLED=false`)
4. Restart server
5. Monitor logs in `logs/` directory

**Zero breaking changes** - fully backward compatible!

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/hshadab/zkx402/issues
- Documentation: https://github.com/hshadab/zkx402
