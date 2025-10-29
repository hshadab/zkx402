# Simple Threshold Model - Comprehensive Test Results

**Model:** simple_threshold.onnx  
**Purpose:** Basic balance check - proves amount < balance  
**Test Date:** October 29, 2025

## Model Logic

The simple_threshold model checks: `amount < balance`

- **Output = 1** → Transaction APPROVED (amount is less than balance)
- **Output = 0** → Transaction DENIED (amount >= balance)

## Test Results

### ✅ PASS Scenarios (Amount < Balance)

| Test # | Amount | Balance | Output | Result | Description |
|--------|--------|---------|--------|--------|-------------|
| 1 | 1000 | 100000 | 1 | ✅ APPROVED | Small amount, large balance |
| 2 | 5000 | 10000 | 1 | ✅ APPROVED | Amount within limit |
| 3 | 0 | 1000 | 1 | ✅ APPROVED | Zero amount (always passes) |
| 4 | 1 | 2 | 1 | ✅ APPROVED | Edge case: minimal difference |

### ❌ DENY Scenarios (Amount >= Balance)

| Test # | Amount | Balance | Output | Result | Description |
|--------|---------|---------|--------|---------|-------------|
| 1 | 10000 | 10000 | 0 | ❌ DENIED | Amount equals balance |
| 2 | 15000 | 10000 | 0 | ❌ DENIED | Amount exceeds balance |
| 3 | 50000 | 10000 | 0 | ❌ DENIED | Large excess over balance |
| 4 | 1000 | 0 | 0 | ❌ DENIED | Amount > 0, balance = 0 |

## Performance Metrics

- **Operations:** 6 ops
- **Proof Generation Time:** ~6-7 seconds
- **Trace Length:** 6
- **Matrix Dimensions:** T=8, K=65536
- **Rows:** 512
- **Cols:** 1024

## Model Behavior Summary

### Expected APPROVAL Conditions:
✅ amount < balance (strict less than)
✅ Handles zero amounts correctly (0 < any positive balance)
✅ Handles very large balances
✅ Consistent behavior across all input ranges

### Expected DENIAL Conditions:
❌ amount == balance (equal amounts are denied)
❌ amount > balance (insufficient funds)
❌ Works correctly even with large differences

## Integration Points

### UI Integration
- Model card displays example scenarios
- Interactive testing available at https://zk-x402.com
- Real-time proof generation with visual feedback

### API Integration
```bash
# Approve scenario
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{"model": "simple_threshold", "inputs": {"amount": "5000", "balance": "10000"}}'

# Deny scenario
curl -X POST http://localhost:3001/api/generate-proof \
  -H "Content-Type: application/json" \
  -d '{"model": "simple_threshold", "inputs": {"amount": "15000", "balance": "10000"}}'
```

### x402 Protocol Integration
- Price: 0.01 USDC per proof
- Payment: USDC on Base L2
- Wallet: 0x1f409E94684804e5158561090Ced8941B47B0CC6
- Full x402 discovery at: https://zk-x402.com/.well-known/x402

## Verification Status

✅ **FULLY VERIFIED AND PRODUCTION READY**

- All approve scenarios tested and working
- All deny scenarios tested and working
- zkML proof generation confirmed
- API endpoints functional
- x402 payment integration complete
- UI examples deployed

## Use Cases

1. **Basic Wallet Checks** - Verify user has sufficient balance before transaction
2. **Spending Limits** - Ensure transactions don't exceed available funds
3. **Payment Authorization** - Prove authorization without revealing actual balance
4. **Agent Payments** - x402 agents can prove they have funds without sharing private data

## Next Steps for Users

1. **Test in UI**: Visit https://zk-x402.com and select "Simple Threshold"
2. **Try API**: Use the curl commands above to generate proofs
3. **Integrate x402**: Follow PAYMENT_GUIDE.md for production integration
4. **Pay per proof**: 0.01 USDC per proof on Base L2

