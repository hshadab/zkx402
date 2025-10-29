#!/bin/bash
BINARY="./jolt-atlas-fork/target/release/examples/proof_json_output"
MODEL="./policy-examples/onnx/curated/simple_threshold.onnx"

echo "=== SIMPLE THRESHOLD MODEL - COMPREHENSIVE TESTING ==="
echo ""

echo "Test 1: APPROVE - Small amount, large balance (1000 < 100000)"
$BINARY "$MODEL" 1000 100000 | tail -1 | jq -r '.approved, .output'
echo ""

echo "Test 2: APPROVE - Equal amounts (5000 < 10000)"
$BINARY "$MODEL" 5000 10000 | tail -1 | jq -r '.approved, .output'
echo ""

echo "Test 3: DENY - Amount equals balance (10000 = 10000)"
$BINARY "$MODEL" 10000 10000 | tail -1 | jq -r '.approved, .output'
echo ""

echo "Test 4: DENY - Amount exceeds balance (15000 > 10000)"
$BINARY "$MODEL" 15000 10000 | tail -1 | jq -r '.approved, .output'
echo ""

echo "Test 5: DENY - Large excess (50000 > 10000)"
$BINARY "$MODEL" 50000 10000 | tail -1 | jq -r '.approved, .output'
echo ""

echo "Test 6: APPROVE - Zero amount (0 < 1000)"
$BINARY "$MODEL" 0 1000 | tail -1 | jq -r '.approved, .output'
echo ""

echo "=== TEST SUMMARY ==="
echo "All tests completed"
