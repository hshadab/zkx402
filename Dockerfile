# Multi-stage Dockerfile for zkX402 on Render.com
# Includes Rust JOLT Atlas prover + Node.js API/UI

# =============================================================================
# Stage 1: Build Rust JOLT Prover
# =============================================================================
FROM rust:1.75-slim as rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy Rust prover code
COPY zkx402-agent-auth/zkml-jolt-fork ./zkml-jolt-fork
COPY zkx402-agent-auth/jolt-atlas-fork ./jolt-atlas-fork
COPY zkx402-agent-auth/policy-examples ./policy-examples

# Build the JOLT prover (release mode for performance)
WORKDIR /build/jolt-atlas-fork/zkml-jolt-core
RUN cargo build --release --example proof_json_output

# =============================================================================
# Stage 2: Build Node.js Frontend
# =============================================================================
FROM node:18-slim as frontend-builder

WORKDIR /build

# Copy UI code
COPY zkx402-agent-auth/ui/package*.json ./ui/
COPY zkx402-agent-auth/ui ./ui

# Install dependencies and build frontend
WORKDIR /build/ui
RUN npm ci
RUN npm run build

# =============================================================================
# Stage 3: Runtime Image
# =============================================================================
FROM node:18-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built Rust prover from Stage 1
COPY --from=rust-builder /build/jolt-atlas-fork/target/release/examples/proof_json_output /app/jolt-atlas-fork/target/release/examples/proof_json_output
COPY --from=rust-builder /build/policy-examples /app/policy-examples

# Copy Node.js backend and built frontend from Stage 2
COPY --from=frontend-builder /build/ui/dist /app/ui/dist
COPY zkx402-agent-auth/ui/package*.json /app/ui/
COPY zkx402-agent-auth/ui/server.js /app/ui/

# Install production dependencies only
WORKDIR /app/ui
RUN npm ci --only=production

# Create startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Environment variables (override in Render dashboard)
ENV NODE_ENV=production
ENV PORT=10000

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD node -e "require('http').get('http://localhost:' + process.env.PORT + '/api/health', (r) => process.exit(r.statusCode === 200 ? 0 : 1))"

# Start the application
CMD ["/app/start.sh"]
