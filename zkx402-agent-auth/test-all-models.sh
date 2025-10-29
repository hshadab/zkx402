#!/bin/bash
# Test all 14 zkx402 models with approve and deny scenarios

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:3001/api/generate-proof"
RESULTS_FILE="test-results-$(date +%Y%m%d-%H%M%S).txt"

echo "========================================" | tee "$RESULTS_FILE"
echo "zkX402 Model Testing - All 14 Models" | tee -a "$RESULTS_FILE"
echo "Testing approve and deny scenarios" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

passed=0
failed=0

test_model() {
    local model_id="$1"
    local model_name="$2"
    local inputs="$3"
    local expected="$4"
    local scenario="$5"

    echo -e "${BLUE}Testing: $model_name - $scenario${NC}"
    echo "Model: $model_name - $scenario" >> "$RESULTS_FILE"
    echo "Inputs: $inputs" >> "$RESULTS_FILE"

    # Make API call
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model_id\", \"inputs\": $inputs}")

    # Check for errors
    if echo "$response" | grep -q "\"error\""; then
        error_msg=$(echo "$response" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
        echo -e "${RED}✗ FAILED: $error_msg${NC}"
        echo "✗ FAILED: $error_msg" >> "$RESULTS_FILE"
        ((failed++))
        echo "" >> "$RESULTS_FILE"
        return 1
    fi

    # Extract approved status
    approved=$(echo "$response" | grep -o '"approved":[^,}]*' | cut -d':' -f2 | tr -d ' ')

    # Verify expected result
    if [ "$approved" == "$expected" ]; then
        proof_time=$(echo "$response" | grep -o '"proofTime":[^,}]*' | cut -d':' -f2)
        echo -e "${GREEN}✓ PASSED (approved=$approved, time=${proof_time}ms)${NC}"
        echo "✓ PASSED (approved=$approved, time=${proof_time}ms)" >> "$RESULTS_FILE"
        ((passed++))
    else
        echo -e "${RED}✗ FAILED: Expected $expected, got $approved${NC}"
        echo "✗ FAILED: Expected $expected, got $approved" >> "$RESULTS_FILE"
        ((failed++))
    fi

    echo "" >> "$RESULTS_FILE"
    sleep 1  # Small delay between tests
}

echo -e "${YELLOW}========== BASIC MODELS (3) ==========${NC}" | tee -a "$RESULTS_FILE"

# 1. Simple Threshold
test_model "simple_threshold" "Simple Threshold" \
    '{"amount": "5000", "balance": "10000"}' "true" "Approve: Sufficient balance"
test_model "simple_threshold" "Simple Threshold" \
    '{"amount": "15000", "balance": "10000"}' "false" "Deny: Insufficient balance"

# 2. Percentage Limit
test_model "percentage_limit" "Percentage Limit" \
    '{"amount": "5000", "balance": "100000", "max_percentage": "10"}' "true" "Approve: 5% (within 10%)"
test_model "percentage_limit" "Percentage Limit" \
    '{"amount": "15000", "balance": "100000", "max_percentage": "10"}' "false" "Deny: 15% (exceeds 10%)"

# 3. Vendor Trust
test_model "vendor_trust" "Vendor Trust" \
    '{"vendor_trust": "75", "min_trust": "50"}' "true" "Approve: High trust vendor"
test_model "vendor_trust" "Vendor Trust" \
    '{"vendor_trust": "30", "min_trust": "50"}' "false" "Deny: Low trust vendor"

echo -e "${YELLOW}========== VELOCITY MODELS (3) ==========${NC}" | tee -a "$RESULTS_FILE"

# 4. Hourly Velocity
test_model "velocity_1h" "Hourly Velocity" \
    '{"amount": "5000", "spent_1h": "10000", "limit_1h": "20000"}' "true" "Approve: Within hourly limit"
test_model "velocity_1h" "Hourly Velocity" \
    '{"amount": "15000", "spent_1h": "10000", "limit_1h": "20000"}' "false" "Deny: Exceeds hourly limit"

# 5. Daily Velocity
test_model "velocity_24h" "Daily Velocity" \
    '{"amount": "5000", "spent_24h": "20000", "limit_24h": "50000"}' "true" "Approve: Within daily limit"
test_model "velocity_24h" "Daily Velocity" \
    '{"amount": "40000", "spent_24h": "20000", "limit_24h": "50000"}' "false" "Deny: Exceeds daily limit"

# 6. Daily Cap
test_model "daily_limit" "Daily Cap" \
    '{"amount": "10000", "daily_spent": "5000", "daily_cap": "20000"}' "true" "Approve: Within daily cap"
test_model "daily_limit" "Daily Cap" \
    '{"amount": "20000", "daily_spent": "5000", "daily_cap": "20000"}' "false" "Deny: Exceeds daily cap"

echo -e "${YELLOW}========== ACCESS MODELS (1) ==========${NC}" | tee -a "$RESULTS_FILE"

# 7. Age Gate
test_model "age_gate" "Age Gate" \
    '{"age": "25", "min_age": "21"}' "true" "Approve: Adult over 21"
test_model "age_gate" "Age Gate" \
    '{"age": "18", "min_age": "21"}' "false" "Deny: Under age limit"

echo -e "${YELLOW}========== ADVANCED MODELS (3) ==========${NC}" | tee -a "$RESULTS_FILE"

# 8. Multi-Factor
test_model "multi_factor" "Multi-Factor" \
    '{"amount": "5000", "balance": "100000", "spent_24h": "20000", "limit_24h": "50000", "vendor_trust": "75", "min_trust": "50"}' "true" "Approve: All checks pass"
test_model "multi_factor" "Multi-Factor" \
    '{"amount": "5000", "balance": "3000", "spent_24h": "20000", "limit_24h": "50000", "vendor_trust": "75", "min_trust": "50"}' "false" "Deny: Insufficient balance"

# 9. Composite Scoring
test_model "composite_scoring" "Composite Scoring" \
    '{"amount": "5000", "balance": "100000", "vendor_trust": "75", "user_history": "80"}' "true" "Approve: High composite score"
test_model "composite_scoring" "Composite Scoring" \
    '{"amount": "5000", "balance": "6000", "vendor_trust": "20", "user_history": "30"}' "false" "Deny: Low composite score"

# 10. Risk Neural
test_model "risk_neural" "Risk Neural" \
    '{"amount": "5000", "balance": "100000", "velocity_1h": "5000", "velocity_24h": "20000", "vendor_trust": "75"}' "true" "Approve: Low risk"
test_model "risk_neural" "Risk Neural" \
    '{"amount": "50000", "balance": "60000", "velocity_1h": "15000", "velocity_24h": "80000", "vendor_trust": "30"}' "false" "Deny: High risk"

echo -e "${YELLOW}========== TEST MODELS (4) ==========${NC}" | tee -a "$RESULTS_FILE"

# 11. Test Less
test_model "test_less" "Test: Less Operation" \
    '{"value_a": "5", "value_b": "10"}' "true" "Approve: 5 < 10"
test_model "test_less" "Test: Less Operation" \
    '{"value_a": "10", "value_b": "5"}' "false" "Deny: 10 !< 5"

# 12. Test Identity
test_model "test_identity" "Test: Identity Operation" \
    '{"value": "42"}' "true" "Test: Identity(42)"
test_model "test_identity" "Test: Identity Operation" \
    '{"value": "100"}' "true" "Test: Identity(100)"

# 13. Test Clip
test_model "test_clip" "Test: Clip Operation" \
    '{"value": "5", "min": "0", "max": "10"}' "true" "Test: Clip(5, 0, 10)"
test_model "test_clip" "Test: Clip Operation" \
    '{"value": "15", "min": "0", "max": "10"}' "true" "Test: Clip(15, 0, 10)"

# 14. Test Slice
test_model "test_slice" "Test: Slice Operation" \
    '{"start": "0", "end": "3"}' "true" "Test: Slice[0:3]"
test_model "test_slice" "Test: Slice Operation" \
    '{"start": "2", "end": "5"}' "true" "Test: Slice[2:5]"

# Summary
echo "" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "TEST SUMMARY" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo -e "Total Tests: $((passed + failed))" | tee -a "$RESULTS_FILE"
echo -e "${GREEN}Passed: $passed${NC}" | tee -a "$RESULTS_FILE"
echo "Passed: $passed" >> "$RESULTS_FILE"
echo -e "${RED}Failed: $failed${NC}" | tee -a "$RESULTS_FILE"
echo "Failed: $failed" >> "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC}" | tee -a "$RESULTS_FILE"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}" | tee -a "$RESULTS_FILE"
    exit 1
fi
