/**
 * Agent-Friendly API Routes for zkX402
 * Provides machine-readable policy metadata for autonomous agents
 */

const { CURATED_MODELS } = require('./x402-middleware');
const { formatPrice } = require('./base-payment');
const webhookManager = require('./webhook-manager');

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

  // ========== AGENT API: POLICY SIMULATION ==========
  app.post('/api/policies/:id/simulate', async (req, res) => {
    const { id } = req.params;
    const { inputs } = req.body;

    const model = CURATED_MODELS[id];

    if (!model) {
      return res.status(404).json({
        error: 'Policy not found',
        policy_id: id
      });
    }

    // Validate inputs
    const missingInputs = model.inputs.filter(input => !(input in inputs));
    if (missingInputs.length > 0) {
      return res.status(400).json({
        error: 'Missing required inputs',
        missing: missingInputs,
        required: model.inputs
      });
    }

    try {
      // Use ONNX Runtime to evaluate model (faster than JOLT proof generation)
      const onnx = require('onnxruntime-node');
      const fs = require('fs');
      const path = require('path');

      const modelPath = path.join(__dirname, '..', 'policy-examples', 'onnx', model.file);

      // Load and run ONNX model
      const session = await onnx.InferenceSession.create(modelPath);

      // Prepare input tensor (using float32 as ONNX models expect float)
      const inputTensor = new onnx.Tensor('float32',
        model.inputs.map(inp => parseFloat(inputs[inp]) || 0),
        [1, model.inputs.length]
      );

      // Run inference
      const feeds = {};
      feeds[session.inputNames[0]] = inputTensor;
      const output = await session.run(feeds);
      const result = output[session.outputNames[0]].data[0];

      const approved = result === 1;

      res.json({
        simulation: true,
        approved,
        output: result,
        policy_id: id,
        policy_name: model.name,
        inputs,
        execution_time_ms: '<1ms (simulation only)',
        note: 'This is a simulation without zkML proof. Use /api/generate-proof to get a verifiable proof.',
        proof_generation: {
          endpoint: `${BASE_URL}/api/generate-proof`,
          estimated_time: model.proofTime,
          estimated_cost: formatPrice(model.price)
        }
      });
    } catch (error) {
      console.error('Simulation error:', error);
      res.status(500).json({
        error: 'Simulation failed',
        details: error.message,
        policy_id: id
      });
    }
  });

  // ========== WEBHOOK API: REGISTER WEBHOOKS ==========
  app.post('/api/webhooks', (req, res) => {
    const { callback_url, metadata } = req.body;

    if (!callback_url) {
      return res.status(400).json({ error: 'callback_url is required' });
    }

    const webhookId = `wh_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const webhook = webhookManager.registerWebhook(webhookId, callback_url, metadata);

    res.status(201).json({
      webhook_id: webhook.id,
      callback_url: webhook.callbackUrl,
      metadata: webhook.metadata,
      created_at: webhook.createdAt,
      status: 'active',
    });
  });

  // ========== WEBHOOK API: GET WEBHOOK ==========
  app.get('/api/webhooks/:id', (req, res) => {
    const { id } = req.params;
    const webhook = webhookManager.getWebhook(id);

    if (!webhook) {
      return res.status(404).json({ error: 'Webhook not found' });
    }

    res.json({
      webhook_id: webhook.id,
      callback_url: webhook.callbackUrl,
      metadata: webhook.metadata,
      created_at: webhook.createdAt,
      deliveries: webhook.deliveries.length,
      recent_deliveries: webhook.deliveries.slice(-5),
    });
  });

  // ========== WEBHOOK API: DELETE WEBHOOK ==========
  app.delete('/api/webhooks/:id', (req, res) => {
    const { id } = req.params;
    const deleted = webhookManager.deleteWebhook(id);

    if (!deleted) {
      return res.status(404).json({ error: 'Webhook not found' });
    }

    res.json({ message: 'Webhook deleted successfully', webhook_id: id });
  });

  // ========== WEBHOOK API: LIST WEBHOOKS ==========
  app.get('/api/webhooks', (req, res) => {
    const webhooks = webhookManager.listWebhooks();

    res.json({
      webhooks: webhooks.map(w => ({
        webhook_id: w.id,
        callback_url: w.callbackUrl,
        created_at: w.createdAt,
        deliveries: w.deliveries.length,
      })),
    });
  });
}

module.exports = { registerAgentRoutes, setBaseUrl };
