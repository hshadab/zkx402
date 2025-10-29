# Curated Authorization Models for zkX402

This directory contains 14 curated ONNX models (10 production + 4 test) for agent authorization using zkX402 + x402. All models are tested, validated, and optimized for JOLT Atlas zero-knowledge proofs.

## Model Categories

### 🛡️ Basic Thresholds (Simple, Fast)
- `simple_threshold` - Amount vs balance check
- `percentage_limit` - Amount as % of balance
- `vendor_trust` - Vendor reputation threshold

### ⏱️ Velocity Limits (Rate Control)
- `velocity_1h` - 1-hour spending rate limit
- `velocity_24h` - 24-hour spending rate limit
- `daily_limit` - Daily spending cap

### 🔐 Access Control (Identity)
- `age_gate` - Age verification for restricted content

### 🎯 Multi-Factor (Advanced)
- `multi_factor` - Combines amount + velocity + trust
- `composite_scoring` - Weighted scoring system
- `risk_neural` - Lightweight neural network

---

## Model Specifications

### 1. simple_threshold.onnx
**Purpose**: Basic authorization - approve if amount < balance

**Use Cases**:
- Wallet spending authorization
- Basic overdraft protection
- Simple balance checks

**Inputs**:
- `amount` (int32): Transaction amount in cents
- `balance` (int32): Current balance in cents

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = (balance - amount) > 0`

**Operations**: Sub, Greater
**Size**: ~30 operations, <50 parameters
**Proof Time**: ~2-3 seconds

**Example**:
```python
inputs = {"amount": 5000, "balance": 10000}  # $50 / $100
output = 1  # Approved
```

---

### 2. percentage_limit.onnx
**Purpose**: Approve if amount is less than X% of balance

**Use Cases**:
- Prevent large withdrawals
- Risk management (max 10% per transaction)
- Spending safety limits

**Inputs**:
- `amount` (int32): Transaction amount in cents
- `balance` (int32): Current balance in cents
- `max_percentage` (int32): Maximum % allowed (e.g., 10 = 10%)

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = (amount * 100) < (balance * max_percentage)`

**Operations**: Mul, Less
**Size**: ~40 operations, <60 parameters
**Proof Time**: ~2-3 seconds

**Example**:
```python
inputs = {"amount": 5000, "balance": 100000, "max_percentage": 10}
# $50 < (10% of $1000) = $50 < $100 → Approved
```

---

### 3. vendor_trust.onnx
**Purpose**: Approve based on vendor reputation score

**Use Cases**:
- Marketplace safety
- Scam prevention
- Trust-based authorization

**Inputs**:
- `vendor_trust` (int32): Vendor reputation score (0-100)
- `min_trust` (int32): Minimum required trust (default: 50)

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = vendor_trust >= min_trust`

**Operations**: GreaterEqual
**Size**: ~20 operations, <30 parameters
**Proof Time**: ~1-2 seconds

**Example**:
```python
inputs = {"vendor_trust": 75, "min_trust": 50}
# 75 >= 50 → Approved
```

---

### 4. velocity_1h.onnx
**Purpose**: Limit spending rate within 1 hour

**Use Cases**:
- Fraud detection
- Rate limiting
- Unusual activity prevention

**Inputs**:
- `amount` (int32): Current transaction amount
- `spent_1h` (int32): Amount spent in past hour
- `limit_1h` (int32): Maximum hourly spend limit

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = (spent_1h + amount) <= limit_1h`

**Operations**: Add, LessEqual (via Less + Equal)
**Size**: ~50 operations, <80 parameters
**Proof Time**: ~2-3 seconds

**Example**:
```python
inputs = {"amount": 5000, "spent_1h": 20000, "limit_1h": 50000}
# $50 + $200 = $250 < $500 → Approved
```

---

### 5. velocity_24h.onnx
**Purpose**: Limit spending rate within 24 hours

**Use Cases**:
- Daily spending caps
- Budget enforcement
- Long-term rate limiting

**Inputs**:
- `amount` (int32): Current transaction amount
- `spent_24h` (int32): Amount spent in past 24 hours
- `limit_24h` (int32): Maximum daily spend limit

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = (spent_24h + amount) <= limit_24h`

**Operations**: Add, LessEqual
**Size**: ~50 operations, <80 parameters
**Proof Time**: ~2-3 seconds

**Example**:
```python
inputs = {"amount": 10000, "spent_24h": 50000, "limit_24h": 100000}
# $100 + $500 = $600 < $1000 → Approved
```

---

### 6. daily_limit.onnx
**Purpose**: Simple daily spending cap

**Use Cases**:
- Allowance enforcement
- Budget management
- Spending control

**Inputs**:
- `amount` (int32): Transaction amount
- `daily_spent` (int32): Amount spent today
- `daily_cap` (int32): Maximum allowed per day

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = (daily_spent + amount) <= daily_cap`

**Operations**: Add, LessEqual
**Size**: ~45 operations, <70 parameters
**Proof Time**: ~2-3 seconds

---

### 7. age_gate.onnx
**Purpose**: Age verification for restricted content

**Use Cases**:
- Alcohol purchases
- Adult content access
- Gambling restrictions
- Age-restricted goods

**Inputs**:
- `age` (int32): User's age in years
- `min_age` (int32): Minimum required age (default: 18 or 21)

**Output**:
- `approved` (int32): 1 if authorized, 0 if rejected

**Logic**: `approved = age >= min_age`

**Operations**: GreaterEqual
**Size**: ~25 operations, <40 parameters
**Proof Time**: ~1-2 seconds

**Example**:
```python
inputs = {"age": 25, "min_age": 21}
# 25 >= 21 → Approved (can buy alcohol)
```

---

### 8. multi_factor.onnx
**Purpose**: Combines amount, velocity, and vendor trust checks

**Use Cases**:
- Comprehensive fraud prevention
- High-value transaction security
- Multi-layered authorization

**Inputs**:
- `amount` (int32): Transaction amount
- `balance` (int32): Current balance
- `spent_24h` (int32): Amount spent in 24 hours
- `limit_24h` (int32): Daily spending limit
- `vendor_trust` (int32): Vendor reputation (0-100)
- `min_trust` (int32): Minimum trust threshold

**Output**:
- `approved` (int32): 1 if all checks pass, 0 otherwise

**Logic**:
```
check1 = (balance - amount) > 0
check2 = (spent_24h + amount) <= limit_24h
check3 = vendor_trust >= min_trust
approved = check1 AND check2 AND check3
```

**Operations**: Sub, Add, Greater, LessEqual, GreaterEqual, Mul (for AND)
**Size**: ~120 operations, ~180 parameters
**Proof Time**: ~4-5 seconds

**Example**:
```python
inputs = {
    "amount": 10000,
    "balance": 50000,
    "spent_24h": 20000,
    "limit_24h": 50000,
    "vendor_trust": 75,
    "min_trust": 50
}
# All checks pass → Approved
```

---

### 9. composite_scoring.onnx
**Purpose**: Weighted scoring system for transaction risk

**Use Cases**:
- Risk-based pricing
- Dynamic authorization thresholds
- Reputation-weighted limits

**Inputs**:
- `amount` (int32): Transaction amount
- `balance` (int32): Current balance
- `vendor_trust` (int32): Vendor score (0-100)
- `user_history` (int32): User reliability score (0-100)

**Output**:
- `risk_score` (int32): Combined risk score (0-100, higher = safer)
- `approved` (int32): 1 if score > threshold, 0 otherwise

**Logic**:
```
balance_score = (balance - amount) / 100  # Remaining balance factor
trust_factor = vendor_trust / 2
history_factor = user_history / 2
risk_score = balance_score + trust_factor + history_factor
approved = risk_score > 50
```

**Operations**: Sub, Div, Add, Greater
**Size**: ~90 operations, ~140 parameters
**Proof Time**: ~3-4 seconds

---

### 10. risk_neural.onnx
**Purpose**: Lightweight neural network for transaction risk prediction

**Use Cases**:
- ML-based fraud detection
- Anomaly detection
- Predictive authorization

**Inputs**:
- `amount` (int32): Transaction amount (scaled)
- `balance` (int32): Current balance (scaled)
- `velocity_1h` (int32): 1-hour spending velocity
- `velocity_24h` (int32): 24-hour spending velocity
- `vendor_trust` (int32): Vendor reputation

**Output**:
- `risk_score` (int32): Neural network risk prediction (0-100)
- `approved` (int32): 1 if safe, 0 if risky

**Architecture**:
```
Input [5] → Hidden [10] → Hidden [5] → Output [1]
Activations: ReLU (approximated via Greater + Mul)
```

**Operations**: MatMult, Add, Greater, Mul (ReLU approximation)
**Size**: ~150 operations, ~200 parameters
**Proof Time**: ~5-6 seconds

**Training**: Pre-trained on synthetic transaction data with known fraud patterns

---

## Test Models

### 11. test_less.onnx
**Purpose**: Test Less comparison operation

**Use Cases**:
- JOLT Atlas operation validation
- Comparison testing
- Basic operation verification

**Inputs**:
- `value1` (int32): First value
- `value2` (int32): Second value

**Output**:
- `result` (int32): 1 if value1 < value2, 0 otherwise

**Logic**: `result = value1 < value2`

**Operations**: Less
**Size**: ~10 operations, <20 parameters
**Proof Time**: ~1 second

---

### 12. test_identity.onnx
**Purpose**: Test Identity pass-through operation

**Use Cases**:
- Graph construction testing
- Residual connections
- Operation verification

**Inputs**:
- `input` (int32): Input value

**Output**:
- `output` (int32): Same as input (pass-through)

**Logic**: `output = identity(input)`

**Operations**: Identity
**Size**: ~5 operations, <10 parameters
**Proof Time**: ~0.5 seconds

---

### 13. test_clip.onnx
**Purpose**: Test Clip operation (ReLU approximation)

**Use Cases**:
- Activation function testing
- ReLU validation
- Neural network component testing

**Inputs**:
- `input` (int32): Input value

**Output**:
- `output` (int32): Clipped value (max(0, input))

**Logic**: `output = clip(input, min=0)`

**Operations**: Clip
**Size**: ~8 operations, <15 parameters
**Proof Time**: ~0.8 seconds

---

### 14. test_slice.onnx
**Purpose**: Test Slice tensor operation

**Use Cases**:
- Feature extraction testing
- Tensor manipulation validation
- Subset selection verification

**Inputs**:
- `tensor` (int32 array): Input tensor
- `start` (int32): Start index
- `end` (int32): End index

**Output**:
- `sliced` (int32 array): Tensor slice from start to end

**Logic**: `sliced = tensor[start:end]`

**Operations**: Slice
**Size**: ~12 operations, <25 parameters
**Proof Time**: ~1 second

---

## Usage Instructions

### API Integration

```javascript
// Use built-in model
const response = await fetch('https://api.zkx402.ai/generate-proof', {
  method: 'POST',
  body: JSON.stringify({
    model: 'multi_factor',  // Use curated model
    inputs: {
      amount: 5000,
      balance: 100000,
      spent_24h: 20000,
      limit_24h: 50000,
      vendor_trust: 75,
      min_trust: 50
    }
  })
});

const { approved, zkmlProof } = await response.json();
```

### Local Testing

```bash
# Test with JOLT prover
cd jolt-prover
./target/release/examples/proof_json_output \
  ../policy-examples/onnx/curated/multi_factor.onnx \
  5000 100000 20000 50000 75 50
```

### Model Selection Guide

| Use Case | Recommended Model | Speed | Security |
|----------|------------------|-------|----------|
| Basic wallet check | `simple_threshold` | ⚡⚡⚡ | 🛡️ |
| Marketplace purchase | `vendor_trust` | ⚡⚡⚡ | 🛡️🛡️ |
| High-value transaction | `multi_factor` | ⚡⚡ | 🛡️🛡️🛡️ |
| Fraud detection | `velocity_1h` | ⚡⚡ | 🛡️🛡️🛡️ |
| ML-based risk | `risk_neural` | ⚡ | 🛡️🛡️🛡️🛡️ |
| Age-restricted goods | `age_gate` | ⚡⚡⚡ | 🛡️🛡️ |

---

## Testing & Validation

All models have been:
- ✅ Validated against JOLT Atlas operation whitelist
- ✅ Tested with sample inputs
- ✅ Size-checked (< MAX_TENSOR_SIZE)
- ✅ Proof generation verified
- ✅ Integer scaling confirmed

## Model Updates

Models are versioned using semantic versioning:
- `v1.0.0` - Initial release
- Future updates will maintain backward compatibility
- Breaking changes will increment major version

---

## License

These curated models are open-source and available under MIT License for use with zkX402 + x402 ecosystem.

## Contributing

To propose new curated models:
1. Follow JOLT Atlas operation constraints
2. Test with JOLT prover
3. Document inputs/outputs clearly
4. Submit PR with model + generator script
5. Include test cases

---

**Last Updated**: 2025-10-28
**Model Version**: v1.0.0
**Compatible with**: JOLT Atlas (zkX402 fork), x402 protocol
