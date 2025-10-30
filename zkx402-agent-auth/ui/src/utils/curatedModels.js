/**
 * Curated ONNX Authorization Models for zkX402
 * Production-ready models covering common x402 payment authorization use cases
 *
 * ✅ JOLT Atlas Status: Production Ready - All Critical Bugs Fixed (2025-10-29)
 * - Gather heap address collision: FIXED
 * - Two-pass input allocation: IMPLEMENTED
 * - Constant index Gather addressing: FIXED
 * - All 14 models (10 production + 4 test) verified and working
 * - Full proof generation and verification successful
 *
 * Verified Models:
 * - 10 production authorization policy models
 * - 4 test models for operation verification
 */

export const CURATED_MODELS = [
  // ========== BASIC THRESHOLDS ==========
  {
    id: 'simple_threshold',
    name: 'Simple Threshold',
    category: 'Basic',
    description: 'Checks if you have enough money',
    useCase: 'Basic wallet balance checks in x402 payments',
    inputs: ['amount', 'balance'],
    operations: '~10 ops',
    proofTime: '~2-3s',
    color: 'accent-green',
    supported: true,
    featured: false,  // Trivial if-then logic
    production: true,  // Available in API but not featured
    examples: [
      { amount: '5000', balance: '10000', expected: 'approved', desc: 'Sufficient balance' },
      { amount: '15000', balance: '10000', expected: 'denied', desc: 'Insufficient balance' }
    ]
  },
  {
    id: 'percentage_limit',
    name: 'Percentage Limit',
    category: 'Basic',
    description: 'Limits spending to X% of balance',
    useCase: '"Don\'t spend more than 10% at once" policies',
    inputs: ['amount', 'balance', 'max_percentage'],
    operations: '~15 ops',
    proofTime: '~7s',
    color: 'accent-green',
    supported: true,  // ✅ Working after Gather fix (2025-10-29)
    featured: true,   // Has actual computation (multiplication + division)
    production: true,
    examples: [
      { amount: '5000', balance: '100000', max_percentage: '10', expected: 'approved', desc: '5% of balance (within 10%)' },
      { amount: '15000', balance: '100000', max_percentage: '10', expected: 'denied', desc: '15% of balance (exceeds 10%)' }
    ]
  },
  {
    id: 'vendor_trust',
    name: 'Vendor Trust',
    category: 'Basic',
    description: 'Requires minimum vendor reputation',
    useCase: 'x402 marketplace transactions',
    inputs: ['vendor_trust', 'min_trust'],
    operations: '~5 ops',
    proofTime: '~2-3s',
    color: 'accent-green',
    supported: true,
    featured: false,  // Trivial comparison
    production: true,
    examples: [
      { vendor_trust: '75', min_trust: '50', expected: 'approved', desc: 'High trust vendor' },
      { vendor_trust: '30', min_trust: '50', expected: 'denied', desc: 'Low trust vendor' }
    ]
  },

  // ========== VELOCITY LIMITS ==========
  {
    id: 'velocity_1h',
    name: 'Hourly Velocity',
    category: 'Velocity',
    description: 'Limits spending per hour',
    useCase: 'Rate limiting, fraud prevention for x402 agents',
    inputs: ['amount', 'spent_1h', 'limit_1h'],
    operations: '~10 ops',
    proofTime: '~2-3s',
    color: 'accent-blue',
    supported: true,
    featured: false,  // Simple addition + comparison
    production: true,
    examples: [
      { amount: '5000', spent_1h: '10000', limit_1h: '20000', expected: 'approved', desc: 'Within hourly limit' },
      { amount: '15000', spent_1h: '10000', limit_1h: '20000', expected: 'denied', desc: 'Exceeds hourly limit' }
    ]
  },
  {
    id: 'velocity_24h',
    name: 'Daily Velocity',
    category: 'Velocity',
    description: 'Limits spending per day',
    useCase: 'Daily spending caps for x402 payments',
    inputs: ['amount', 'spent_24h', 'limit_24h'],
    operations: '~10 ops',
    proofTime: '~2-3s',
    color: 'accent-blue',
    supported: true,
    featured: false,  // Duplicate of velocity_1h
    production: true,
    examples: [
      { amount: '5000', spent_24h: '20000', limit_24h: '50000', expected: 'approved', desc: 'Within daily limit' },
      { amount: '40000', spent_24h: '20000', limit_24h: '50000', expected: 'denied', desc: 'Exceeds daily limit' }
    ]
  },
  {
    id: 'daily_limit',
    name: 'Daily Cap',
    category: 'Velocity',
    description: 'Hard cap on daily spending',
    useCase: 'Budget enforcement for agent spending',
    inputs: ['amount', 'daily_spent', 'daily_cap'],
    operations: '~10 ops',
    proofTime: '~2-3s',
    color: 'accent-blue',
    supported: true,
    featured: false,  // Duplicate of velocity checks
    production: true,
    examples: [
      { amount: '10000', daily_spent: '5000', daily_cap: '20000', expected: 'approved', desc: 'Within daily cap' },
      { amount: '20000', daily_spent: '5000', daily_cap: '20000', expected: 'denied', desc: 'Exceeds daily cap' }
    ]
  },

  // ========== ACCESS CONTROL ==========
  {
    id: 'age_gate',
    name: 'Age Gate',
    category: 'Access',
    description: 'Checks minimum age requirement',
    useCase: 'Age-restricted x402 purchases',
    inputs: ['age', 'min_age'],
    operations: '~5 ops',
    proofTime: '~2-3s',
    color: 'accent-purple',
    supported: true,
    featured: false,  // Trivial comparison
    production: true,
    examples: [
      { age: '25', min_age: '21', expected: 'approved', desc: 'Adult over 21' },
      { age: '18', min_age: '21', expected: 'denied', desc: 'Under age limit' }
    ]
  },

  // ========== ADVANCED ==========
  {
    id: 'multi_factor',
    name: 'Multi-Factor',
    category: 'Advanced',
    description: 'Combines balance + velocity + trust',
    useCase: 'High-security x402 transactions',
    inputs: ['amount', 'balance', 'spent_24h', 'limit_24h', 'vendor_trust', 'min_trust'],
    operations: '~30 ops',
    proofTime: '~2-3s',
    color: 'accent-orange',
    supported: true,
    featured: true,   // ⭐ Combines multiple checks - shows composition
    production: true,
    examples: [
      {
        amount: '5000',
        balance: '100000',
        spent_24h: '20000',
        limit_24h: '50000',
        vendor_trust: '75',
        min_trust: '50',
        expected: 'approved',
        desc: 'All checks pass'
      },
      {
        amount: '5000',
        balance: '3000',
        spent_24h: '20000',
        limit_24h: '50000',
        vendor_trust: '75',
        min_trust: '50',
        expected: 'denied',
        desc: 'Insufficient balance'
      }
    ]
  },
  {
    id: 'composite_scoring',
    name: 'Composite Scoring',
    category: 'Advanced',
    description: 'Weighted risk score from multiple factors',
    useCase: 'Advanced risk assessment for x402',
    inputs: ['amount', 'balance', 'vendor_trust', 'user_history'],
    operations: '72 ops',
    proofTime: '9.3s',
    color: 'accent-orange',
    supported: true,  // ✅ Working - Verified 72 operations (2025-10-29)
    featured: true,   // ⭐ 72 ops - weighted scoring model, real computation
    production: true,
    examples: [
      {
        amount: '5000',
        balance: '100000',
        vendor_trust: '75',
        user_history: '80',
        expected: 'approved',
        desc: 'High composite score'
      },
      {
        amount: '5000',
        balance: '6000',
        vendor_trust: '20',
        user_history: '30',
        expected: 'denied',
        desc: 'Low composite score'
      }
    ]
  },
  {
    id: 'risk_neural',
    name: 'Risk Neural',
    category: 'Advanced',
    description: 'ML-based risk scoring',
    useCase: 'Sophisticated fraud detection for agent payments',
    inputs: ['amount', 'balance', 'velocity_1h', 'velocity_24h', 'vendor_trust'],
    operations: '~47 ops',
    proofTime: '~8s',
    color: 'accent-orange',
    supported: true,  // ✅ Working after Gather fix (2025-10-29)
    featured: true,   // ⭐ 47 ops - ACTUAL NEURAL NETWORK - killer feature!
    production: true,
    examples: [
      {
        amount: '5000',
        balance: '100000',
        velocity_1h: '5000',
        velocity_24h: '20000',
        vendor_trust: '75',
        expected: 'approved',
        desc: 'Low risk transaction'
      },
      {
        amount: '50000',
        balance: '60000',
        velocity_1h: '15000',
        velocity_24h: '80000',
        vendor_trust: '30',
        expected: 'denied',
        desc: 'High risk transaction'
      }
    ]
  },

  // ========== TEST MODELS (Operation Verification) ==========
  {
    id: 'test_less',
    name: 'Test: Less Operation',
    category: 'Test',
    description: 'Verifies Less (<) comparison operation',
    useCase: 'Testing that comparisons work correctly in zkML proofs',
    inputs: ['value_a', 'value_b'],
    operations: '3 ops',
    proofTime: '~4s',
    color: 'accent-gray',
    supported: true,
    featured: false,  // Internal testing only
    production: false,  // Don't show in UI
    examples: [
      { value_a: '5', value_b: '10', expected: 'true', desc: '5 < 10 = true' },
      { value_a: '10', value_b: '5', expected: 'false', desc: '10 < 5 = false' }
    ]
  },
  {
    id: 'test_identity',
    name: 'Test: Identity Operation',
    category: 'Test',
    description: 'Verifies Identity pass-through operation',
    useCase: 'Testing that values pass through unchanged in zkML',
    inputs: ['value'],
    operations: '2 ops',
    proofTime: '~4s',
    color: 'accent-gray',
    supported: true,
    featured: false,  // Internal testing only
    production: false,  // Don't show in UI
    examples: [
      { value: '42', expected: '42', desc: 'Identity(42) = 42' },
      { value: '100', expected: '100', desc: 'Identity(100) = 100' }
    ]
  },
  {
    id: 'test_clip',
    name: 'Test: Clip Operation',
    category: 'Test',
    description: 'Verifies Clip/ReLU activation function',
    useCase: 'Testing neural network activation functions in zkML',
    inputs: ['value', 'min', 'max'],
    operations: '3 ops',
    proofTime: '~4s',
    color: 'accent-gray',
    supported: true,
    featured: false,  // Internal testing only
    production: false,  // Don't show in UI
    examples: [
      { value: '5', min: '0', max: '10', expected: '5', desc: 'Clip(5, 0, 10) = 5' },
      { value: '15', min: '0', max: '10', expected: '10', desc: 'Clip(15, 0, 10) = 10' }
    ]
  },
  {
    id: 'test_slice',
    name: 'Test: Slice Operation',
    category: 'Test',
    description: 'Verifies Slice tensor operation',
    useCase: 'Testing array slicing operations in zkML proofs',
    inputs: ['start', 'end'],
    operations: '4 ops',
    proofTime: '~4.5s',
    color: 'accent-gray',
    supported: true,
    featured: false,  // Internal testing only
    production: false,  // Don't show in UI
    examples: [
      { start: '0', end: '3', expected: '[0,1,2]', desc: 'Slice tensor [0:3]' },
      { start: '2', end: '5', expected: '[2,3,4]', desc: 'Slice tensor [2:5]' }
    ]
  }
]

/**
 * Input field configurations with labels and descriptions
 */
export const INPUT_CONFIGS = {
  amount: {
    label: 'Transaction Amount',
    description: 'Amount to authorize',
    placeholder: '50.00',
    example: '$50.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  balance: {
    label: 'Account Balance',
    description: 'Current balance',
    placeholder: '100.00',
    example: '$100.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  max_percentage: {
    label: 'Maximum Percentage',
    description: 'Max % of balance to spend',
    placeholder: '10',
    example: '10%',
    type: 'number',
    min: 0,
    max: 100
  },
  vendor_trust: {
    label: 'Vendor Trust Score',
    description: 'Vendor reputation (0-100)',
    placeholder: '75',
    example: '75',
    type: 'number',
    min: 0,
    max: 100
  },
  min_trust: {
    label: 'Minimum Trust Required',
    description: 'Minimum acceptable trust score',
    placeholder: '50',
    example: '50',
    type: 'number',
    min: 0,
    max: 100
  },
  spent_1h: {
    label: '1-Hour Spending',
    description: 'Amount spent in last hour',
    placeholder: '100.00',
    example: '$100.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  limit_1h: {
    label: '1-Hour Limit',
    description: 'Maximum allowed per hour',
    placeholder: '200.00',
    example: '$200.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  spent_24h: {
    label: '24-Hour Spending',
    description: 'Amount spent today',
    placeholder: '200.00',
    example: '$200.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  limit_24h: {
    label: '24-Hour Limit',
    description: 'Maximum allowed per day',
    placeholder: '500.00',
    example: '$500.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  daily_spent: {
    label: 'Daily Spending',
    description: 'Amount spent today',
    placeholder: '50.00',
    example: '$50.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  daily_cap: {
    label: 'Daily Cap',
    description: 'Maximum daily spending',
    placeholder: '200.00',
    example: '$200.00',
    type: 'number',
    min: 0,
    step: 0.01
  },
  age: {
    label: 'User Age',
    description: 'Age in years',
    placeholder: '25',
    example: '25 years',
    type: 'number',
    min: 0,
    max: 150
  },
  min_age: {
    label: 'Minimum Age',
    description: 'Required minimum age',
    placeholder: '21',
    example: '21 years',
    type: 'number',
    min: 0,
    max: 150
  },
  user_history: {
    label: 'User History Score',
    description: 'Historical behavior score (0-100)',
    placeholder: '80',
    example: '80',
    type: 'number',
    min: 0,
    max: 100
  },
  // Test model inputs
  value_a: {
    label: 'Value A',
    description: 'First value for comparison',
    placeholder: '5',
    example: '5',
    type: 'number'
  },
  value_b: {
    label: 'Value B',
    description: 'Second value for comparison',
    placeholder: '10',
    example: '10',
    type: 'number'
  },
  value: {
    label: 'Input Value',
    description: 'Value to test',
    placeholder: '42',
    example: '42',
    type: 'number'
  },
  min: {
    label: 'Minimum Value',
    description: 'Lower bound for clipping',
    placeholder: '0',
    example: '0',
    type: 'number'
  },
  max: {
    label: 'Maximum Value',
    description: 'Upper bound for clipping',
    placeholder: '10',
    example: '10',
    type: 'number'
  },
  start: {
    label: 'Start Index',
    description: 'Starting index for slice',
    placeholder: '0',
    example: '0',
    type: 'number',
    min: 0
  },
  end: {
    label: 'End Index',
    description: 'Ending index for slice',
    placeholder: '3',
    example: '3',
    type: 'number',
    min: 0
  }
}

/**
 * Model categories for organization
 */
export const MODEL_CATEGORIES = {
  Basic: { color: 'accent-green', description: 'Simple threshold checks' },
  Velocity: { color: 'accent-blue', description: 'Rate limiting and spending caps' },
  Access: { color: 'accent-purple', description: 'Access control policies' },
  Advanced: { color: 'accent-orange', description: 'Multi-factor and ML-based' },
  Test: { color: 'accent-gray', description: 'Operation verification tests' }
}

/**
 * Get only supported models
 */
export function getSupportedModels() {
  return CURATED_MODELS.filter(m => m.supported !== false)
}

/**
 * Get featured models (advanced, complex ones for UI)
 */
export function getFeaturedModels() {
  return CURATED_MODELS.filter(m => m.featured === true)
}

/**
 * Get production models (excludes test models)
 */
export function getProductionModels() {
  return CURATED_MODELS.filter(m => m.production !== false)
}

/**
 * Get model by ID
 */
export function getModel(id) {
  return CURATED_MODELS.find(m => m.id === id)
}

/**
 * Get models by category
 */
export function getModelsByCategory(category) {
  return CURATED_MODELS.filter(m => m.category === category)
}

/**
 * Get all model categories
 */
export function getAllCategories() {
  return Object.keys(MODEL_CATEGORIES)
}

/**
 * Get input configuration
 */
export function getInputConfig(inputName) {
  return INPUT_CONFIGS[inputName] || {
    label: inputName,
    description: '',
    placeholder: '',
    type: 'text'
  }
}

/**
 * Format value for display
 */
export function formatInputValue(inputName, value) {
  if (inputName.includes('amount') || inputName.includes('balance') || inputName.includes('spent') || inputName.includes('limit') || inputName.includes('cap')) {
    return `$${(parseInt(value) / 100).toFixed(2)}`
  }
  if (inputName.includes('trust') || inputName.includes('percentage') || inputName.includes('history')) {
    return `${value}%`
  }
  if (inputName.includes('age')) {
    return `${value} years`
  }
  return value
}
