#!/bin/bash
# Startup script for zkX402 on Render.com

set -e

echo "=========================================="
echo "zkX402 Agent Authorization"
echo "Starting on Render.com..."
echo "=========================================="

# Use Render's PORT environment variable
export PORT="${PORT:-10000}"

echo "Port: $PORT"
echo "Node environment: $NODE_ENV"
echo "Models directory: /app/policy-examples/onnx"
echo "JOLT prover: /app/jolt-prover/target/release/examples/proof_json_output"
echo ""

# Verify critical files exist
if [ ! -f "/app/jolt-prover/target/release/examples/proof_json_output" ]; then
    echo "ERROR: JOLT prover binary not found!"
    exit 1
fi

if [ ! -d "/app/policy-examples/onnx" ]; then
    echo "ERROR: ONNX models directory not found!"
    exit 1
fi

echo "✓ JOLT prover binary found"
echo "✓ ONNX models directory found"
echo ""

# List available models
echo "Available ONNX models:"
ls -lh /app/policy-examples/onnx/*.onnx 2>/dev/null || echo "  (none found)"
echo ""

# Start the Node.js server
echo "Starting API server on port $PORT..."
cd /app/ui
exec node server.js
