# zkX402 Deployment Guide

Complete guide for deploying zkX402 Agent Authorization to production.

## Table of Contents

- [Deployment Options](#deployment-options)
- [Render.com Deployment](#rendercom-deployment) ⭐ **Recommended**
- [Docker Deployment](#docker-deployment)
- [Railway Deployment](#railway-deployment)
- [Manual Server Deployment](#manual-server-deployment)
- [Production Checklist](#production-checklist)
- [Monitoring](#monitoring)

## Deployment Options

| Option | Difficulty | Cost | Best For |
|--------|------------|------|----------|
| **Render.com** ⭐ | **Easy** | **$25-85/mo** | **Production-ready, auto-deploy, persistent storage** |
| Docker | Easy | Free-$ | Development, small scale |
| Railway | Easy | $5-20/mo | Quick production deployment |
| VPS (Digital Ocean, AWS) | Medium | $10-50/mo | Full control, scaling |
| Kubernetes | Hard | $$$ | Enterprise, high availability |

---

## Render.com Deployment

**✅ Recommended for production deployment of zkX402 with real JOLT Atlas proofs.**

Render.com provides:
- ✅ Automatic Docker builds with Rust + Node.js
- ✅ Persistent storage for model uploads
- ✅ Auto-deploy on git push
- ✅ Free SSL certificates
- ✅ Health monitoring and auto-restart
- ✅ Easy scaling and rollbacks

### Quick Deploy

**Option 1: One-Click Deploy Button**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/hshadab/zkx402)

**Option 2: Blueprint Deploy**

```bash
# 1. Ensure deployment files are committed
git add Dockerfile render.yaml start.sh .dockerignore
git commit -m "Add Render deployment config"
git push origin main

# 2. Go to https://dashboard.render.com
# 3. Click "New" → "Blueprint"
# 4. Connect your GitHub repo
# 5. Render auto-detects render.yaml and deploys
```

**Option 3: Manual Setup**

```bash
# 1. Log in to Render: https://dashboard.render.com
# 2. Click "New" → "Web Service"
# 3. Connect GitHub repository
# 4. Configure:
#    - Name: zkx402-agent-auth
#    - Environment: Docker
#    - Dockerfile Path: ./Dockerfile
#    - Instance Type: Standard or higher
#    - Branch: main
# 5. Add environment variables:
#    NODE_ENV=production
# 6. Set health check path: /api/health
# 7. Click "Create Web Service"
```

### Build Time

- **First build**: ~10-15 minutes (compiles Rust prover)
- **Subsequent builds**: ~5-8 minutes (cached layers)

### Performance

**Proof Generation Times** (Standard instance):
- Simple Auth: ~1.5-2.0s
- Neural Auth: ~2.5-3.5s

**Instance Recommendations**:
- **Standard** ($25/mo): Good for testing
- **Pro** ($85/mo): Recommended for production
- **Pro Plus** ($185/mo): High traffic applications

### Complete Documentation

📚 **See [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md) for complete guide including:**
- Step-by-step deployment instructions
- Configuration options
- Scaling and monitoring
- Troubleshooting
- Cost optimization
- Security best practices

---

## Docker Deployment

### 1. Create Dockerfile

Create `zkx402-agent-auth/ui/Dockerfile`:

```dockerfile
FROM rust:1.75 as rust-builder

WORKDIR /app

# Copy Rust workspace
COPY zkx402-agent-auth/jolt-prover ./jolt-prover
COPY zkx402-agent-auth/jolt-atlas-fork ./jolt-atlas-fork
COPY zkx402-agent-auth/zkml-jolt-fork ./zkml-jolt-fork
COPY zkx402-agent-auth/dory-fork ./dory-fork

# Build Rust prover
WORKDIR /app/jolt-prover
RUN cargo build --release --example proof_json_output

# Node.js stage
FROM node:20 as node-builder

WORKDIR /app/ui

# Copy UI code
COPY zkx402-agent-auth/ui/package*.json ./
RUN npm ci --only=production

COPY zkx402-agent-auth/ui .
RUN npm run build

# Final stage
FROM node:20-slim

WORKDIR /app

# Install Python for ONNX model generation
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX dependencies
RUN pip3 install torch numpy onnx onnxruntime --break-system-packages

# Copy Rust binaries
COPY --from=rust-builder /app/jolt-prover/target/release/examples/proof_json_output /app/jolt-prover/
COPY --from=rust-builder /app/jolt-prover /app/jolt-prover

# Copy Node.js app
COPY --from=node-builder /app/ui/node_modules ./node_modules
COPY zkx402-agent-auth/ui/server.js ./
COPY zkx402-agent-auth/ui/dist ./dist

# Copy ONNX models
COPY zkx402-agent-auth/policy-examples/onnx/*.onnx ./models/

EXPOSE 3001

ENV NODE_ENV=production
ENV PORT=3001

CMD ["node", "server.js"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  zkx402:
    build: ./zkx402-agent-auth/ui
    ports:
      - "3001:3001"
    environment:
      - NODE_ENV=production
      - PORT=3001
    volumes:
      - ./zkx402-agent-auth/policy-examples/onnx:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - zkx402
    restart: unless-stopped
```

### 3. Build and Run

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Check logs
docker-compose logs -f zkx402

# Stop
docker-compose down
```

### 4. Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream zkx402 {
        server zkx402:3001;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            return 301 https://$server_name$request_uri;
        }
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Rate limiting
        limit_req_zone $binary_remote_addr zone=proof_limit:10m rate=10r/m;

        location /api/generate-proof {
            limit_req zone=proof_limit burst=5 nodelay;
            proxy_pass http://zkx402;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 180s;
        }

        location / {
            proxy_pass http://zkx402;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

## Railway Deployment

### 1. Install Railway CLI

```bash
npm install -g @railway/cli
railway login
```

### 2. Create railway.json

```json
{
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "cd zkx402-agent-auth/jolt-prover && cargo build --release --example proof_json_output && cd ../ui && npm install && npm run build"
  },
  "deploy": {
    "startCommand": "cd zkx402-agent-auth/ui && node server.js",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

### 3. Deploy

```bash
railway init
railway up
railway open
```

### 4. Configure Environment Variables

In Railway dashboard, add:
```
NODE_ENV=production
PORT=3001
```

## Manual Server Deployment (Ubuntu 22.04)

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python
sudo apt install -y python3 python3-pip
pip3 install torch numpy onnx onnxruntime

# Install build essentials
sudo apt install -y build-essential pkg-config libssl-dev
```

### 2. Clone and Build

```bash
# Clone repository
git clone https://github.com/yourusername/zkx402.git
cd zkx402/zkx402-agent-auth

# Generate ONNX models
cd policy-examples/onnx
python3 create_demo_models.py
cd ../..

# Build Rust prover
cd jolt-prover
cargo build --release --example proof_json_output
cd ..

# Install and build UI
cd ui
npm install
npm run build
cd ..
```

### 3. Create Systemd Service

Create `/etc/systemd/system/zkx402.service`:

```ini
[Unit]
Description=zkX402 Agent Authorization Service
After=network.target

[Service]
Type=simple
User=zkx402
WorkingDirectory=/home/zkx402/zkx402/zkx402-agent-auth/ui
Environment="NODE_ENV=production"
Environment="PORT=3001"
ExecStart=/usr/bin/node server.js
Restart=on-failure
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=zkx402

[Install]
WantedBy=multi-user.target
```

### 4. Start Service

```bash
# Create user
sudo useradd -r -s /bin/false zkx402
sudo chown -R zkx402:zkx402 /home/zkx402/zkx402

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable zkx402
sudo systemctl start zkx402

# Check status
sudo systemctl status zkx402

# View logs
sudo journalctl -u zkx402 -f
```

### 5. Setup Nginx Reverse Proxy

```bash
sudo apt install -y nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/zkx402

# (Use nginx.conf from Docker section above)

# Enable site
sudo ln -s /etc/nginx/sites-available/zkx402 /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. SSL with Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Production Checklist

### Security

- [ ] Enable HTTPS/SSL
- [ ] Implement API authentication (JWT, API keys)
- [ ] Add rate limiting
- [ ] Set up CORS properly
- [ ] Validate all inputs
- [ ] Use environment variables for secrets
- [ ] Enable security headers

```javascript
// Add to server.js
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

app.use(helmet());

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100
});
app.use('/api/', limiter);
```

### Performance

- [ ] Enable proof caching
- [ ] Use production Node.js mode
- [ ] Enable gzip compression
- [ ] Set up CDN for static assets
- [ ] Configure connection pooling
- [ ] Optimize Rust release build

### Monitoring

- [ ] Set up logging (Winston, Pino)
- [ ] Configure error tracking (Sentry)
- [ ] Add health checks
- [ ] Monitor proof generation metrics
- [ ] Set up uptime monitoring
- [ ] Configure alerts

### Backup

- [ ] Backup ONNX models
- [ ] Backup configuration
- [ ] Document recovery procedures

## Monitoring

### Health Check Endpoint

```bash
# Check service health
curl https://your-domain.com/api/health

# Expected response
{
  "status": "ok",
  "timestamp": "2025-10-28T12:00:00.000Z",
  "modelsAvailable": 5
}
```

### Logging

Add structured logging:

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});

// Log proof generation
logger.info('Proof generated', {
  model: 'simple_auth',
  approved: true,
  duration: '750ms'
});
```

### Metrics

Track key metrics:

- Proof generation requests/minute
- Success/failure rate
- Average proof generation time
- Model usage distribution
- Error types and frequency

### Alerts

Set up alerts for:

- Service downtime
- High error rate (>5%)
- Slow proof generation (>5s)
- Disk space low
- High CPU usage

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml with scaling
services:
  zkx402:
    build: ./zkx402-agent-auth/ui
    deploy:
      replicas: 3
    environment:
      - NODE_ENV=production

  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf
```

### Vertical Scaling

- CPU: 4+ cores recommended
- RAM: 8GB+ for optimal performance
- Storage: 20GB+ SSD

## Troubleshooting

### Service Won't Start

```bash
# Check logs
sudo journalctl -u zkx402 -n 50

# Check port availability
sudo netstat -tulpn | grep 3001

# Test manually
cd /home/zkx402/zkx402/zkx402-agent-auth/ui
node server.js
```

### Slow Proof Generation

- Check CPU usage: `htop`
- Ensure Rust binary is release build
- Consider caching proofs
- Scale horizontally

### Out of Memory

- Increase Node.js heap: `NODE_OPTIONS="--max-old-space-size=4096"`
- Add swap space
- Scale to larger instance

## Support

- **Issues**: https://github.com/yourusername/zkx402/issues
- **Documentation**: https://github.com/yourusername/zkx402/tree/main/docs

---

**Version**: 1.0.0
**Last Updated**: 2025-10-28
