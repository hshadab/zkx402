# Environment Variables Reference

This document describes all environment variables used by the zkX402 server application.

## Server Configuration

### `PORT`
- **Type**: Number
- **Required**: No
- **Default**: `3001`
- **Description**: HTTP server port
- **Example**: `PORT=3000`

### `NODE_ENV`
- **Type**: String
- **Required**: No
- **Default**: `development`
- **Values**: `development`, `production`, `test`
- **Description**: Application environment mode. Affects CORS policy, logging level, and other behaviors.
- **Example**: `NODE_ENV=production`

---

## Payment Configuration

### `PAYMENT_WALLET`
- **Type**: Ethereum Address (string)
- **Required**: **YES**
- **Default**: None (hardcoded fallback exists but should not be used)
- **Description**: Base L2 wallet address that receives USDC payments
- **Example**: `PAYMENT_WALLET=0x1f409E94684804e5158561090Ced8941B47B0CC6`
- **Security**: This is public information (receiving address), safe to commit to repository

### `PAYMENT_PRIVATE_KEY`
- **Type**: Ethereum Private Key (string, prefixed with 0x)
- **Required**: **YES** (for payment verification)
- **Default**: None
- **Description**: Private key for the payment wallet, used to verify transaction signatures
- **Example**: `PAYMENT_PRIVATE_KEY=0x...` (NEVER commit actual private keys!)
- **Security**: ⚠️ **CRITICAL** - Never commit this to version control. Use secure secret management.

---

## Redis Cache Configuration

### `REDIS_ENABLED`
- **Type**: Boolean (string: 'true' or 'false')
- **Required**: No
- **Default**: `true`
- **Description**: Enable/disable Redis caching for proof results
- **Example**: `REDIS_ENABLED=true`
- **Note**: If disabled, falls back to in-memory caching

### `REDIS_HOST`
- **Type**: String
- **Required**: No (if `REDIS_ENABLED=false`)
- **Default**: `localhost`
- **Description**: Redis server hostname
- **Example**: `REDIS_HOST=redis.example.com`

### `REDIS_PORT`
- **Type**: Number
- **Required**: No
- **Default**: `6379`
- **Description**: Redis server port
- **Example**: `REDIS_PORT=6380`

### `REDIS_PASSWORD`
- **Type**: String
- **Required**: No
- **Default**: None (no authentication)
- **Description**: Redis authentication password (if Redis server requires auth)
- **Example**: `REDIS_PASSWORD=your-secure-password`
- **Security**: Keep secure, don't commit to version control

### `REDIS_TLS`
- **Type**: Boolean (string: 'true' or 'false')
- **Required**: No
- **Default**: `false`
- **Description**: Enable TLS/SSL for Redis connection
- **Example**: `REDIS_TLS=true`

---

## JOLT Prover Configuration

### `PREFER_NO_DIV`
- **Type**: Boolean (string: '1' or '0')
- **Required**: No
- **Default**: `0` (off)
- **Description**: Prefer division-free model variants when available. JOLT Atlas has known issues with Div operations, so models with `_no_div` suffix are more reliable.
- **Example**: `PREFER_NO_DIV=1`
- **Note**: If enabled, server will use `model_no_div.onnx` instead of `model.onnx` when available

### `JOLT_TRACE_TRANSCRIPT`
- **Type**: String
- **Required**: No
- **Default**: None
- **Description**: Enable JOLT transcript tracing for debugging proof generation
- **Example**: `JOLT_TRACE_TRANSCRIPT=1`
- **Use Case**: Debugging prover issues

### `JOLT_TRACE_DIV`
- **Type**: String
- **Required**: No
- **Default**: None
- **Description**: Enable division operation tracing in JOLT
- **Example**: `JOLT_TRACE_DIV=1`
- **Use Case**: Debugging division-related proof failures

### `JOLT_REWRITE_CONST_DIV`
- **Type**: String
- **Required**: No
- **Default**: None
- **Description**: Enable constant division rewriting optimization
- **Example**: `JOLT_REWRITE_CONST_DIV=1`
- **Use Case**: Performance optimization

### `JOLT_DIV_V2`
- **Type**: String
- **Required**: No
- **Default**: None
- **Description**: Use version 2 of division implementation
- **Example**: `JOLT_DIV_V2=1`
- **Use Case**: Alternative division strategy

### `JOLT_SUMCHECK_CHUNK`
- **Type**: String
- **Required**: No
- **Default**: None
- **Description**: Enable sumcheck chunking optimization
- **Example**: `JOLT_SUMCHECK_CHUNK=1`
- **Use Case**: Performance tuning for large proofs

### `JOLT_SUMCHECK_CHUNK_SIZE`
- **Type**: Number (string)
- **Required**: No
- **Default**: None
- **Description**: Sumcheck chunk size for optimization
- **Example**: `JOLT_SUMCHECK_CHUNK_SIZE=1024`
- **Use Case**: Fine-tuning sumcheck performance

### `JOLT_SUMCHECK_BIND_LOW2HIGH`
- **Type**: String
- **Required**: No
- **Default**: None
- **Description**: Sumcheck binding direction optimization
- **Example**: `JOLT_SUMCHECK_BIND_LOW2HIGH=1`
- **Use Case**: Advanced prover tuning

---

## CORS Configuration

### `ALLOWED_ORIGINS`
- **Type**: Comma-separated string
- **Required**: **YES** (for production)
- **Default**:
  - Development: `http://localhost:3000,http://localhost:5173,http://localhost:3001`
  - Production: `[]` (no origins allowed by default)
- **Description**: Allowed CORS origins for cross-origin requests
- **Example**: `ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com`
- **Security**: In production, **must** be explicitly configured. Empty in production = no CORS access.

---

## Example .env Files

### Development (.env.development)

```bash
# Server
PORT=3001
NODE_ENV=development

# Payment (use test wallet for development)
PAYMENT_WALLET=0xYourTestWallet
PAYMENT_PRIVATE_KEY=0xYourTestPrivateKey

# Redis (optional for local dev)
REDIS_ENABLED=false

# JOLT Prover
PREFER_NO_DIV=1

# CORS (permissive for local dev)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Production (.env.production)

```bash
# Server
PORT=3001
NODE_ENV=production

# Payment (REQUIRED)
PAYMENT_WALLET=${SECRET_PAYMENT_WALLET}
PAYMENT_PRIVATE_KEY=${SECRET_PAYMENT_PRIVATE_KEY}

# Redis (recommended for production)
REDIS_ENABLED=true
REDIS_HOST=redis.production.example.com
REDIS_PORT=6379
REDIS_PASSWORD=${SECRET_REDIS_PASSWORD}
REDIS_TLS=true

# JOLT Prover
PREFER_NO_DIV=1

# CORS (REQUIRED - whitelist your domains)
ALLOWED_ORIGINS=https://app.example.com,https://api.example.com
```

---

## Security Best Practices

1. **Never commit `.env` files to version control**
   - Already included in `.gitignore`
   - Use `.env.example` as a template

2. **Use secure secret management in production**
   - Environment variables via deployment platform (Render, Heroku, AWS, etc.)
   - Secret managers (AWS Secrets Manager, HashiCorp Vault, etc.)
   - Never hardcode secrets in code

3. **Rotate credentials regularly**
   - Private keys
   - Redis passwords
   - API keys

4. **Restrict CORS origins in production**
   - Never use `*` or empty array in production
   - Whitelist only trusted domains

5. **Use TLS for Redis in production**
   - Set `REDIS_TLS=true`
   - Use secure passwords

---

## Troubleshooting

### "PAYMENT_WALLET environment variable is required"
- Set `PAYMENT_WALLET` in your `.env` file
- Must be a valid Ethereum address (0x...)

### Redis connection errors
- Check `REDIS_HOST` and `REDIS_PORT` are correct
- Verify Redis server is running
- Check firewall rules
- Try setting `REDIS_ENABLED=false` for local development

### CORS errors in production
- Ensure `ALLOWED_ORIGINS` includes your frontend domain
- Check protocol (http vs https)
- Verify no trailing slashes in origins

### Proof generation fails with division errors
- Set `PREFER_NO_DIV=1` to use division-free model variants
- Check that `_no_div.onnx` files exist in `policy-examples/`

---

## Additional Resources

- [Main README](../README.md)
- [Deployment Guide](../DEPLOYMENT.md)
- [Wallet Setup](../WALLET_SETUP.md)
- [API Reference](../API_REFERENCE.md)
