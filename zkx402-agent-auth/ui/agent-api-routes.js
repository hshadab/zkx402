/**
 * Agent-Friendly API Routes for zkX402
 * Provides machine-readable policy metadata for autonomous agents
 */

const { CURATED_MODELS } = require('./x402-middleware');
const { formatPrice } = require('./base-payment');

// Base URL (will be passed from server.js)
let BASE_URL = 'http://localhost:3001';

function setBaseUrl(url) {
  BASE_URL = url;
}

// Helper function to get example inputs
function getExampleInputs(modelId, scenario) {
  const examples = {
    simple_threshold: {
      approve: { amount: 5000, balance: 10000 },
      deny: { amount: 15000, balance: 10000 }
    },
    percentage_limit: {
      approve: { amount: 5000, balance: 100000, max_percentage: 10 },
      deny: { amount: 15000, balance: 100000, max_percentage: 10 }
    },
    vendor_trust: {
      approve: { vendor_trust: 75, min_trust: 50 },
      deny: { vendor_trust: 30, min_trust: 50 }
    },
    velocity_1h: {
      approve: { amount: 5000, spent_1h: 10000, limit_1h: 20000 },
      deny: { amount: 15000, spent_1h: 10000, limit_1h: 20000 }
    },
    velocity_24h: {
      approve: { amount: 5000, spent_24h: 20000, limit_24h: 50000 },
      deny: { amount: 40000, spent_24h: 20000, limit_24h: 50000 }
    },
    daily_limit: {
      approve: { amount: 10000, daily_spent: 5000, daily_cap: 20000 },
      deny: { amount: 20000, daily_spent: 5000, daily_cap: 20000 }
    },
    age_gate: {
      approve: { age: 25, min_age: 21 },
      deny: { age: 18, min_age: 21 }
    },
    multi_factor: {
      approve: { amount: 5000, balance: 100000, spent_24h: 20000, limit_24h: 50000, vendor_trust: 75, min_trust: 50 },
      deny: { amount: 5000, balance: 3000, spent_24h: 20000, limit_24h: 50000, vendor_trust: 75, min_trust: 50 }
    },
    composite_scoring: {
      approve: { amount: 5000, balance: 100000, vendor_trust: 75, user_history: 80 },
      deny: { amount: 5000, balance: 6000, vendor_trust: 20, user_history: 30 }
    },
    risk_neural: {
      approve: { amount: 5000, balance: 100000, velocity_1h: 5000, velocity_24h: 20000, vendor_trust: 75 },
      deny: { amount: 50000, balance: 60000, velocity_1h: 15000, velocity_24h: 80000, vendor_trust: 30 }
    },
    test_less: {
      approve: { value_a: 5, value_b: 10 },
      deny: { value_a: 10, value_b: 5 }
    },
    test_identity: {
      approve: { value: 42 },
      deny: { value: 100 }
    },
    test_clip: {
      approve: { value: 5, min: 0, max: 10 },
      deny: { value: 15, min: 0, max: 10 }
    },
    test_slice: {
      approve: { start: 0, end: 3 },
      deny: { start: 2, end: 5 }
    }
  };

  return examples[modelId]?.[scenario] || {};
}

// Helper function to get input descriptions
function getInputDescription(modelId, inputName) {
  const descriptions = {
    amount: 'Transaction amount to authorize',
    balance: 'Current account balance',
    max_percentage: 'Maximum percentage of balance allowed',
    vendor_trust: 'Vendor reputation score (0-100)',
    min_trust: 'Minimum required trust score',
    spent_1h: 'Amount spent in last hour',
    limit_1h: 'Hourly spending limit',
    spent_24h: 'Amount spent in last 24 hours',
    limit_24h: 'Daily spending limit',
    daily_spent: 'Amount spent today',
    daily_cap: 'Maximum daily spending allowed',
    age: 'User age in years',
    min_age: 'Minimum required age',
    user_history: 'User history score (0-100)',
    velocity_1h: 'Spending velocity in last hour',
    velocity_24h: 'Spending velocity in last 24 hours',
    value_a: 'First value for comparison',
    value_b: 'Second value for comparison',
    value: 'Input value',
    min: 'Minimum clip value',
    max: 'Maximum clip value',
    start: 'Slice start index',
    end: 'Slice end index'
  };

  return descriptions[inputName] || `Input parameter: ${inputName}`;
}

// Register routes with Express app
function registerAgentRoutes(app) {

  // ========== AGENT API: POLICIES LISTING ==========
  app.get('/api/policies', (req, res) => {
    const policies = Object.keys(CURATED_MODELS).map(id => {
      const model = CURATED_MODELS[id];
      return {
        id,
        name: model.name,
        description: model.description,
        category: model.category,
        complexity: model.operations <= 10 ? 'simple' : model.operations <= 30 ? 'medium' : 'advanced',
        operations: model.operations,
        avg_proof_time_ms: parseFloat(model.proofTime.replace('~', '').replace('s', '')) * 1000,
        price_usdc: formatPrice(model.price),
        price_atomic: model.price,

        schema: {
          inputs: model.inputs.reduce((acc, input) => {
            acc[input] = {
              type: 'int32',
              description: getInputDescription(id, input),
              required: true
            };
            return acc;
          }, {}),
          output: {
            type: 'int32',
            description: '1 = approved, 0 = denied'
          }
        },

        example: {
          approve: getExampleInputs(id, 'approve'),
          deny: getExampleInputs(id, 'deny')
        },

        use_case: model.useCase,
        endpoint: `${BASE_URL}/x402/authorize/${id}`,
        schema_url: `${BASE_URL}/api/policies/${id}/schema`
      };
    });

    res.json({
      version: '1.3.0',
      total_policies: policies.length,
      policies,
      capabilities: {
        max_model_params: 1024,
        max_operations: 100,
        supported_types: ['int8', 'int16', 'int32', 'float32']
      }
    });
  });

  // ========== AGENT API: POLICY SCHEMA ==========
  app.get('/api/policies/:id/schema', (req, res) => {
    const { id } = req.params;
    const model = CURATED_MODELS[id];

    if (!model) {
      return res.status(404).json({ error: 'Policy not found' });
    }

    res.json({
      policy_id: id,
      name: model.name,
      description: model.description,
      category: model.category,
      version: '1.0.0',

      schema: {
        inputs: model.inputs.reduce((acc, input) => {
          acc[input] = {
            type: 'int32',
            description: getInputDescription(id, input),
            required: true,
            validation: {
              type: 'integer',
              minimum: 0
            }
          };
          return acc;
        }, {}),

        output: {
          type: 'int32',
          description: '1 = approved (authorization granted), 0 = denied (authorization rejected)',
          values: {
            1: 'approved',
            0: 'denied'
          }
        }
      },

      examples: {
        approve: {
          description: 'Example that will be approved',
          inputs: getExampleInputs(id, 'approve'),
          expected_output: 1
        },
        deny: {
          description: 'Example that will be denied',
          inputs: getExampleInputs(id, 'deny'),
          expected_output: 0
        }
      },

      pricing: {
        price_usdc: formatPrice(model.price),
        price_atomic: model.price,
        currency: 'USDC',
        network: 'base'
      },

      performance: {
        operations: model.operations,
        avg_proof_time: model.proofTime,
        complexity: model.operations <= 10 ? 'simple' : model.operations <= 30 ? 'medium' : 'advanced'
      },

      usage: {
        endpoint: `${BASE_URL}/api/generate-proof`,
        method: 'POST',
        request_format: {
          model: id,
          inputs: model.inputs.reduce((acc, input) => {
            acc[input] = '<value>';
            return acc;
          }, {})
        }
      }
    });
  });
}

module.exports = { registerAgentRoutes, setBaseUrl };
