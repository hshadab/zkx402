#!/usr/bin/env bash
# Build JOLT Atlas binary locally and commit it to skip Render compilation
# This reduces build time from 15 minutes to ~2 minutes

set -e

echo "========================================="
echo "Build and Commit JOLT Atlas Binary"
echo "========================================="

cd "$(dirname "$0")/../jolt-atlas-fork/zkml-jolt-core"

echo "🔨 Building JOLT Atlas proof_json_output binary..."
echo "⏰ This will take 10-15 minutes on first build"
echo ""

cargo build --release --example proof_json_output

BINARY_PATH="../../jolt-atlas-fork/target/release/examples/proof_json_output"

if [ -f "$BINARY_PATH" ]; then
    echo ""
    echo "✅ Binary built successfully!"
    echo "📦 Size: $(ls -lh "$BINARY_PATH" | awk '{print $5}')"
    echo ""

    echo "📝 Checking if .gitignore needs updating..."
    cd ../..

    # Remove target/ from .gitignore if present
    if grep -q "^target/$" .gitignore 2>/dev/null; then
        echo "⚠️  Found 'target/' in .gitignore"
        echo "   You need to modify .gitignore to allow committing the binary:"
        echo ""
        echo "   Option 1: Add exception after 'target/' line:"
        echo "     target/"
        echo "     !jolt-atlas-fork/target/release/examples/proof_json_output"
        echo ""
        echo "   Option 2: Comment out 'target/' and add specific ignores"
    fi

    echo ""
    echo "📤 Adding binary to git..."
    git add jolt-atlas-fork/target/release/examples/proof_json_output

    echo ""
    echo "✅ Binary staged for commit!"
    echo ""
    echo "Next steps:"
    echo "  1. Commit the binary:"
    echo "     git commit -m 'Add pre-built JOLT Atlas binary for faster Render builds'"
    echo ""
    echo "  2. Push to GitHub:"
    echo "     git push origin main"
    echo ""
    echo "  3. Render will now:"
    echo "     - Skip the 15-minute compilation"
    echo "     - Use the pre-built binary"
    echo "     - Build time: ~2 minutes (just npm install + vite build)"
    echo ""
    echo "⚠️  Remember to rebuild and recommit the binary if you:"
    echo "   - Update JOLT Atlas code"
    echo "   - Change Rust dependencies"
    echo "   - Modify proof generation logic"
    echo ""
else
    echo "❌ ERROR: Binary not found at $BINARY_PATH"
    exit 1
fi
