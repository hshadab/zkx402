#!/usr/bin/env bash
# Render.com build script for zkX402 with JOLT Atlas zkML prover

set -e  # Exit on error

echo "========================================="
echo "zkX402 Render Build Script"
echo "========================================="

# Install Rust if not already installed
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(rustc --version)"
fi

# Ensure cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

# Build Vite frontend
echo "Building Vite frontend..."
npm run build

# Build JOLT Atlas zkML prover binary
echo "Building JOLT Atlas zkML prover..."
cd ../jolt-atlas-fork/zkml-jolt-core

# Build the proof_json_output example
echo "Compiling proof_json_output binary (this may take 10-15 minutes)..."
cargo build --release --example proof_json_output

# Verify binary was created
if [ -f "../../jolt-atlas-fork/target/release/examples/proof_json_output" ]; then
    echo "✅ JOLT Atlas binary built successfully!"
    ls -lh ../../jolt-atlas-fork/target/release/examples/proof_json_output
else
    echo "❌ ERROR: JOLT Atlas binary not found!"
    exit 1
fi

cd ../../ui

echo "========================================="
echo "Build complete!"
echo "========================================="
