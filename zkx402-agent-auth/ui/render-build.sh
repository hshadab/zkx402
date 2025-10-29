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

# Build JOLT Atlas zkML prover binary
BINARY_PATH="../jolt-atlas-fork/target/release/examples/proof_json_output"

# Check if binary already exists (from previous build or committed)
if [ -f "$BINARY_PATH" ]; then
    echo "✅ JOLT Atlas binary already exists!"
    echo "📦 Binary size: $(ls -lh "$BINARY_PATH" | awk '{print $5}')"
    echo "🕐 Last modified: $(stat -c %y "$BINARY_PATH" 2>/dev/null || stat -f %Sm "$BINARY_PATH" 2>/dev/null)"

    # Check if we should rebuild (if Rust source files changed)
    # This is optional - you can force rebuild by deleting the binary
    echo "⏭️  Skipping compilation (binary exists). To force rebuild, delete the binary first."
else
    echo "🔨 Building JOLT Atlas zkML prover..."
    echo "Changing to JOLT directory..."
    cd ../jolt-atlas-fork/zkml-jolt-core || {
        echo "❌ ERROR: Cannot find jolt-atlas-fork directory!"
        echo "Looking for directories..."
        ls -la ../
        exit 1
    }

    # Build the proof_json_output example
    echo "Compiling proof_json_output binary (this may take 10-15 minutes)..."
    echo "⏰ Started at: $(date)"
    cargo build --release --example proof_json_output
    echo "✅ Finished at: $(date)"

    # Verify binary was created
    if [ -f "../../jolt-atlas-fork/target/release/examples/proof_json_output" ]; then
        echo "✅ JOLT Atlas binary built successfully!"
        ls -lh "../../jolt-atlas-fork/target/release/examples/proof_json_output"
    else
        echo "❌ ERROR: JOLT Atlas binary not found!"
        echo "Checking target directory..."
        ls -la ../../jolt-atlas-fork/target/release/ || echo "Target directory not found"
        exit 1
    fi

    cd ../../ui
fi

echo "========================================="
echo "Build complete!"
echo "========================================="
