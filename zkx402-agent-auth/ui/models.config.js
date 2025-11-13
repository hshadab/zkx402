/**
 * Centralized Model Configuration for zkX402
 *
 * All ONNX authorization models with their metadata, pricing, and examples
 *
 * Categories:
 * - 4 Basic: simple_threshold, percentage_limit, vendor_trust, age_gate
 * - 3 Velocity: velocity_1h, velocity_24h, daily_limit
 * - 3 Advanced: multi_factor, composite_scoring, risk_neural
 * - 4 Test: test_less, test_identity, test_clip, test_slice
 */

const CURATED_MODELS = {
  simple_threshold: {
    file: 'curated/simple_threshold.onnx',
    name: 'Simple Threshold',
    description: 'Basic balance check - proves amount < balance',
    category: 'Basic',
    inputs: ['amount', 'balance'],
    price: '1000', // 0.01 USDC (atomic units)
    useCase: 'Basic wallet balance checks',
    operations: 6,
    proofTime: '1-6.5 min',
    provingTime: '6.6s',
    verificationTime: '1-6 min',
    examples: {
      approve: { amount: 5000, balance: 10000 },
      deny: { amount: 15000, balance: 10000 }
    }
  },

  percentage_limit: {
    file: 'curated/percentage_limit_no_div.onnx',
    name: 'Percentage Limit',
    description: 'Limits spending to X% of balance (division-free)',
    category: 'Basic',
    inputs: ['amount', 'balance', 'max_percentage'],
    price: '1500',
    useCase: 'Percentage-based spending limits',
    operations: 20,
    proofTime: '1-6 min',
    provingTime: '~7s',
    verificationTime: '1-6 min',
    examples: {
      approve: { amount: 5000, balance: 100000, max_percentage: 10 },
      deny: { amount: 15000, balance: 100000, max_percentage: 10 }
    }
  },

  vendor_trust: {
    file: 'curated/vendor_trust.onnx',
    name: 'Vendor Trust',
    description: 'Requires minimum vendor reputation score',
    category: 'Basic',
    inputs: ['vendor_trust', 'min_trust'],
    price: '1000',
    useCase: 'Marketplace vendor verification',
    operations: 5,
    proofTime: '1-6 min',
    provingTime: '~5s',
    verificationTime: '1-6 min',
    examples: {
      approve: { vendor_trust: 75, min_trust: 50 },
      deny: { vendor_trust: 30, min_trust: 50 }
    }
  },

  age_gate: {
    file: 'curated/age_gate.onnx',
    name: 'Age Gate',
    description: 'Checks minimum age requirement',
    category: 'Access',
    inputs: ['age', 'min_age'],
    price: '1000',
    useCase: 'Age-restricted purchases',
    operations: 5,
    proofTime: '1-6 min',
    provingTime: '~5s',
    verificationTime: '1-6 min',
    examples: {
      approve: { age: 25, min_age: 21 },
      deny: { age: 18, min_age: 21 }
    }
  },

  velocity_1h: {
    file: 'curated/velocity_1h.onnx',
    name: 'Hourly Velocity',
    description: 'Limits spending per hour',
    category: 'Velocity',
    inputs: ['amount', 'spent_1h', 'limit_1h'],
    price: '2000',
    useCase: 'Hourly rate limiting',
    operations: 8,
    proofTime: '1-5 min',
    provingTime: '6.4s',
    verificationTime: '40s-4 min',
    examples: {
      approve: { amount: 5000, spent_1h: 10000, limit_1h: 20000 },
      deny: { amount: 15000, spent_1h: 10000, limit_1h: 20000 }
    }
  },

  velocity_24h: {
    file: 'curated/velocity_24h.onnx',
    name: 'Daily Velocity',
    description: 'Limits spending per 24 hours',
    category: 'Velocity',
    inputs: ['amount', 'spent_24h', 'limit_24h'],
    price: '2000',
    useCase: 'Daily spending caps',
    operations: 8,
    proofTime: '1-5 min',
    provingTime: '6.2s',
    verificationTime: '40s-4 min',
    examples: {
      approve: { amount: 5000, spent_24h: 20000, limit_24h: 50000 },
      deny: { amount: 40000, spent_24h: 20000, limit_24h: 50000 }
    }
  },

  daily_limit: {
    file: 'curated/daily_limit.onnx',
    name: 'Daily Cap',
    description: 'Hard cap on daily spending',
    category: 'Velocity',
    inputs: ['amount', 'daily_spent', 'daily_cap'],
    price: '2000',
    useCase: 'Budget enforcement',
    operations: 8,
    proofTime: '1-5 min',
    provingTime: '8.4s',
    verificationTime: '1-4 min',
    examples: {
      approve: { amount: 10000, daily_spent: 5000, daily_cap: 20000 },
      deny: { amount: 20000, daily_spent: 5000, daily_cap: 20000 }
    }
  },

  multi_factor: {
    file: 'curated/multi_factor.onnx',
    name: 'Multi-Factor',
    description: 'Combines balance + velocity + trust checks',
    category: 'Advanced',
    inputs: ['amount', 'balance', 'spent_24h', 'limit_24h', 'vendor_trust', 'min_trust'],
    price: '5000',
    useCase: 'High-security transactions',
    operations: 17,
    proofTime: '5-8 min',
    provingTime: '6.2s',
    verificationTime: '5-7.5 min',
    examples: {
      approve: { amount: 5000, balance: 100000, spent_24h: 20000, limit_24h: 50000, vendor_trust: 75, min_trust: 50 },
      deny: { amount: 5000, balance: 3000, spent_24h: 20000, limit_24h: 50000, vendor_trust: 75, min_trust: 50 }
    }
  },

  composite_scoring: {
    file: 'curated/composite_scoring_no_div.onnx',
    name: 'Composite Scoring',
    description: 'Weighted risk score from multiple factors (division-free)',
    category: 'Advanced',
    inputs: ['amount', 'balance', 'vendor_trust', 'user_history'],
    price: '4000',
    useCase: 'Advanced risk assessment',
    operations: 32,
    proofTime: '5-8 min',
    provingTime: '~8s',
    verificationTime: '5-7 min',
    examples: {
      approve: { amount: 5000, balance: 100000, vendor_trust: 75, user_history: 80 },
      deny: { amount: 5000, balance: 6000, vendor_trust: 20, user_history: 30 }
    }
  },

  risk_neural: {
    file: 'curated/risk_neural_no_div.onnx',
    name: 'Risk Neural',
    description: 'ML-based risk scoring with neural network (division-free)',
    category: 'Advanced',
    inputs: ['amount', 'balance', 'velocity_1h', 'velocity_24h', 'vendor_trust'],
    price: '6000',
    useCase: 'Sophisticated fraud detection',
    operations: 46,
    proofTime: '5-8 min',
    provingTime: '~8s',
    verificationTime: '5-7 min',
    examples: {
      approve: { amount: 5000, balance: 100000, velocity_1h: 5000, velocity_24h: 20000, vendor_trust: 75 },
      deny: { amount: 50000, balance: 60000, velocity_1h: 15000, velocity_24h: 80000, vendor_trust: 30 }
    }
  },

  // Test models for operation verification
  test_less: {
    file: 'curated/test_less.onnx',
    name: 'Test: Less Operation',
    description: 'Verifies Less (<) comparison operation',
    category: 'Test',
    inputs: ['value_a', 'value_b'],
    price: '500',
    useCase: 'Operation verification',
    operations: 3,
    proofTime: '1-5 min',
    provingTime: '~4s',
    verificationTime: '1-5 min',
    examples: {
      approve: { value_a: 5, value_b: 10 },
      deny: { value_a: 10, value_b: 5 }
    }
  },

  test_identity: {
    file: 'curated/test_identity.onnx',
    name: 'Test: Identity Operation',
    description: 'Verifies Identity pass-through operation',
    category: 'Test',
    inputs: ['value'],
    price: '500',
    useCase: 'Operation verification',
    operations: 2,
    proofTime: '~4s',
    examples: {
      approve: { value: 42 },
      deny: { value: 100 }
    }
  },

  test_clip: {
    file: 'curated/test_clip.onnx',
    name: 'Test: Clip Operation',
    description: 'Verifies Clip/ReLU activation function',
    category: 'Test',
    inputs: ['value', 'min', 'max'],
    price: '500',
    useCase: 'Operation verification',
    operations: 3,
    proofTime: '~4s',
    examples: {
      approve: { value: 5, min: 0, max: 10 },
      deny: { value: 15, min: 0, max: 10 }
    }
  },

  test_slice: {
    file: 'curated/test_slice.onnx',
    name: 'Test: Slice Operation',
    description: 'Verifies Slice tensor operation',
    category: 'Test',
    inputs: ['start', 'end'],
    price: '500',
    useCase: 'Operation verification',
    operations: 4,
    proofTime: '~4.5s',
    examples: {
      approve: { start: 0, end: 3 },
      deny: { start: 2, end: 5 }
    }
  }
};

/**
 * Input field descriptions for API documentation
 */
const INPUT_DESCRIPTIONS = {
  amount: 'Transaction amount to authorize',
  balance: 'Current account balance',
  max_percentage: 'Maximum percentage of balance allowed (0-100)',
  vendor_trust: 'Vendor reputation score (0-100)',
  min_trust: 'Minimum required trust score',
  spent_1h: 'Amount spent in last hour',
  limit_1h: 'Maximum spending limit per hour',
  spent_24h: 'Amount spent in last 24 hours',
  limit_24h: 'Maximum spending limit per 24 hours',
  daily_spent: 'Amount spent today',
  daily_cap: 'Maximum daily spending cap',
  age: 'User age in years',
  min_age: 'Minimum required age',
  user_history: 'User history score (0-100)',
  velocity_1h: 'Spending velocity in last hour',
  velocity_24h: 'Spending velocity in last 24 hours',
  value: 'Input value for test',
  value_a: 'First value for comparison',
  value_b: 'Second value for comparison',
  min: 'Minimum clip value',
  max: 'Maximum clip value',
  start: 'Start index for slice',
  end: 'End index for slice'
};

module.exports = {
  CURATED_MODELS,
  INPUT_DESCRIPTIONS
};
