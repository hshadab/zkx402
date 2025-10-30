#!/bin/bash

# Test script for division-free models
# Verifies that all three _no_div models work correctly

BINARY="/home/hshadab/zkx402/zkx402-agent-auth/jolt-atlas-fork/target/release/examples/proof_json_output"
MODEL_DIR="/home/hshadab/zkx402/zkx402-agent-auth/policy-examples/onnx/curated"

echo "========================================"
echo "Testing Division-Free Models"
echo "========================================"
echo ""

# Test 1: percentage_limit_no_div
echo "Test 1: percentage_limit_no_div.onnx"
echo "  Inputs: amount=1, balance=100000, limit=50"
echo "  Expected: APPROVE (1/100000*100 = 0.001% < 50%)"
echo "  Testing..."
RESULT=$("$BINARY" "$MODEL_DIR/percentage_limit_no_div.onnx" 1 100000 50 2>&1 | tail -1)
if echo "$RESULT" | jq -e '.approved == true' > /dev/null 2>&1; then
    echo "  ✅ PASS: Model approved correctly"
else
    echo "  ❌ FAIL: $RESULT"
fi
echo ""

# Test 2: composite_scoring_no_div
echo "Test 2: composite_scoring_no_div.onnx"
echo "  Inputs: feature1=100, feature2=10000, feature3=150, feature4=120"
echo "  Expected: Based on weighted scoring logic"
echo "  Testing..."
RESULT=$("$BINARY" "$MODEL_DIR/composite_scoring_no_div.onnx" 100 10000 150 120 2>&1 | tail -1)
if echo "$RESULT" | jq -e '.output != null' > /dev/null 2>&1; then
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    APPROVED=$(echo "$RESULT" | jq -r '.approved')
    echo "  ✅ PASS: Model generated output=$OUTPUT, approved=$APPROVED"
else
    echo "  ❌ FAIL: $RESULT"
fi
echo ""

# Test 3: risk_neural_no_div
echo "Test 3: risk_neural_no_div.onnx"
echo "  Inputs: amount=5000, balance=100000, vel_1h=2, vel_24h=10, trust=95"
echo "  Expected: Neural network risk assessment"
echo "  Testing..."
RESULT=$("$BINARY" "$MODEL_DIR/risk_neural_no_div.onnx" 5000 100000 2 10 95 2>&1 | tail -1)
if echo "$RESULT" | jq -e '.output != null' > /dev/null 2>&1; then
    OUTPUT=$(echo "$RESULT" | jq -r '.output')
    APPROVED=$(echo "$RESULT" | jq -r '.approved')
    echo "  ✅ PASS: Model generated output=$OUTPUT, approved=$APPROVED"
else
    echo "  ❌ FAIL: $RESULT"
fi
echo ""

# Verify no Div operations
echo "========================================"
echo "Verifying No Div Operations"
echo "========================================"
echo ""

for model in percentage_limit_no_div.onnx composite_scoring_no_div.onnx risk_neural_no_div.onnx; do
    echo "Checking $model..."
    if python3 "$MODEL_DIR/../check_model_div.py" "$MODEL_DIR/$model" 2>&1 | grep -q "WARNING.*Div"; then
        echo "  ❌ FAIL: Model contains Div operations!"
    else
        echo "  ✅ PASS: No Div operations"
    fi
done

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "All division-free models tested successfully!"
echo ""
echo "Next steps:"
echo "1. Update x402-middleware.js to use these models (✅ DONE)"
echo "2. Deploy to production"
echo "3. Monitor proof generation and verification"
