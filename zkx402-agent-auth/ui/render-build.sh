#!/usr/bin/env bash
# Render.com build script for zkX402 with JOLT Atlas zkML prover

set -e  # Exit on error

echo "========================================="
echo "zkX402 Render Build Script"
echo "========================================="
echo "Current directory: $(pwd)"
echo "Contents: $(ls -la)"

# Initialize zkml-jolt-fork submodule (required for jolt-atlas-fork)
echo "Initializing zkml-jolt-fork submodule..."
cd ../..
git submodule update --init --recursive zkx402-agent-auth/zkml-jolt-fork
cd zkx402-agent-auth/ui

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

# NOTE: JOLT Atlas prover build is currently disabled due to upstream dependency issues
# See zkx402-agent-auth/archived/X402_TEST_RESULTS.md for details
# The ICME-Lab/zkml-jolt dependency has arkworks compatibility issues that prevent compilation
#
# Skipping prover build - UI will deploy with proof generation unavailable
echo "⚠️  Skipping JOLT Atlas prover build (upstream dependency issues)"
echo "    See archived/X402_TEST_RESULTS.md for details"
echo "    UI will deploy with model showcase functionality"

echo "========================================="
echo "Build complete!"
echo "========================================="
