/**
 * Demo data for zkX402 Agent Authorization
 * Pre-configured test scenarios for different authorization policies
 */

export const DEMO_SCENARIOS = {
  // Scenario 1: Low risk transaction (should approve)
  lowRisk: {
    name: 'Low Risk Transaction',
    description: 'Small transaction from trusted vendor',
    inputs: {
      amount: '500',        // $5.00
      balance: '10000',     // $100.00
      velocity_1h: '200',   // $2.00 spent in last hour
      velocity_24h: '1500', // $15.00 spent in last 24h
      vendor_trust: '80'    // 80% trust score
    },
    expectedResult: 'approved'
  },

  // Scenario 2: High risk transaction (should deny)
  highRisk: {
    name: 'High Risk Transaction',
    description: 'Large transaction from untrusted vendor',
    inputs: {
      amount: '5000',       // $50.00
      balance: '10000',     // $100.00
      velocity_1h: '800',   // $8.00 spent in last hour
      velocity_24h: '2500', // $25.00 spent in last 24h
      vendor_trust: '30'    // 30% trust score
    },
    expectedResult: 'denied'
  },

  // Scenario 3: Medium transaction (borderline)
  mediumRisk: {
    name: 'Medium Risk Transaction',
    description: 'Moderate transaction with mixed signals',
    inputs: {
      amount: '1000',       // $10.00
      balance: '12000',     // $120.00
      velocity_1h: '400',   // $4.00 spent in last hour
      velocity_24h: '1800', // $18.00 spent in last 24h
      vendor_trust: '60'    // 60% trust score
    },
    expectedResult: 'approved'
  },

  // Scenario 4: Velocity limit exceeded
  velocityExceeded: {
    name: 'Velocity Limit Exceeded',
    description: 'Transaction exceeds spending velocity limits',
    inputs: {
      amount: '300',        // $3.00
      balance: '20000',     // $200.00
      velocity_1h: '600',   // $6.00 spent in last hour (exceeds 500 limit)
      velocity_24h: '2100', // $21.00 spent in last 24h (exceeds 2000 limit)
      vendor_trust: '90'    // 90% trust score (high)
    },
    expectedResult: 'denied'
  },

  // Scenario 5: Low trust vendor
  lowTrustVendor: {
    name: 'Low Trust Vendor',
    description: 'Transaction from vendor with low trust score',
    inputs: {
      amount: '200',        // $2.00
      balance: '15000',     // $150.00
      velocity_1h: '100',   // $1.00 spent in last hour
      velocity_24h: '500',  // $5.00 spent in last 24h
      vendor_trust: '40'    // 40% trust score (below 50 threshold)
    },
    expectedResult: 'denied'
  },

  // Scenario 6: High balance transaction
  highBalance: {
    name: 'High Balance Transaction',
    description: 'Large transaction but with very high balance',
    inputs: {
      amount: '2000',       // $20.00
      balance: '50000',     // $500.00
      velocity_1h: '300',   // $3.00 spent in last hour
      velocity_24h: '1200', // $12.00 spent in last 24h
      vendor_trust: '85'    // 85% trust score
    },
    expectedResult: 'approved'
  },

  // Scenario 7: First transaction of the day
  firstTransaction: {
    name: 'First Transaction',
    description: 'First transaction with no recent spending',
    inputs: {
      amount: '800',        // $8.00
      balance: '8000',      // $80.00
      velocity_1h: '0',     // $0.00 spent in last hour
      velocity_24h: '0',    // $0.00 spent in last 24h
      vendor_trust: '75'    // 75% trust score
    },
    expectedResult: 'approved'
  },

  // Scenario 8: Exactly at 10% threshold
  atThreshold: {
    name: 'At Threshold Limit',
    description: 'Transaction exactly at 10% of balance',
    inputs: {
      amount: '1000',       // $10.00
      balance: '10000',     // $100.00 (10% = $10.00)
      velocity_1h: '200',   // $2.00 spent in last hour
      velocity_24h: '800',  // $8.00 spent in last 24h
      vendor_trust: '70'    // 70% trust score
    },
    expectedResult: 'denied' // Should deny as rule is amount < 10% (not <=)
  }
};

/**
 * Get a random demo scenario
 */
export function getRandomScenario() {
  const keys = Object.keys(DEMO_SCENARIOS);
  const randomKey = keys[Math.floor(Math.random() * keys.length)];
  return DEMO_SCENARIOS[randomKey];
}

/**
 * Get scenario by key
 */
export function getScenario(key) {
  return DEMO_SCENARIOS[key];
}

/**
 * Get all scenario names
 */
export function getAllScenarioNames() {
  return Object.entries(DEMO_SCENARIOS).map(([key, scenario]) => ({
    key,
    name: scenario.name
  }));
}

/**
 * Validate input values
 */
export function validateInputs(inputs) {
  const errors = [];

  if (!inputs.amount || parseInt(inputs.amount) < 0) {
    errors.push('Amount must be a positive number');
  }

  if (!inputs.balance || parseInt(inputs.balance) <= 0) {
    errors.push('Balance must be a positive number');
  }

  if (parseInt(inputs.amount) > parseInt(inputs.balance)) {
    errors.push('Amount cannot exceed balance');
  }

  if (!inputs.velocity_1h || parseInt(inputs.velocity_1h) < 0) {
    errors.push('1-hour velocity must be a non-negative number');
  }

  if (!inputs.velocity_24h || parseInt(inputs.velocity_24h) < 0) {
    errors.push('24-hour velocity must be a non-negative number');
  }

  if (!inputs.vendor_trust || parseInt(inputs.vendor_trust) < 0 || parseInt(inputs.vendor_trust) > 100) {
    errors.push('Vendor trust score must be between 0 and 100');
  }

  return errors;
}

/**
 * Format scaled value to currency
 */
export function formatCurrency(scaledValue) {
  return `$${(parseInt(scaledValue) / 100).toFixed(2)}`;
}

/**
 * Format trust score as percentage
 */
export function formatTrustScore(score) {
  return `${score}%`;
}
