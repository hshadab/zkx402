/**
 * x402 Protocol Middleware for zkX402
 * Implements x402 HTTP payment protocol with zkML proof-based authorization
 *
 * Spec: https://github.com/coinbase/x402
 */

const CURATED_MODELS = {
  simple_threshold: {
    file: 'curated/simple_threshold.onnx',
    name: 'Simple Threshold',
    description: 'Basic balance check - proves amount < balance',
    category: 'Basic',
    inputs: ['amount', 'balance'],
    price: '1000', // 0.01 USDC (atomic units)
    useCase: 'Basic wallet balance checks'
  },
  percentage_limit: {
    file: 'curated/percentage_limit.onnx',
    name: 'Percentage Limit',
    description: 'Limits spending to X% of balance',
    category: 'Basic',
    inputs: ['amount', 'balance', 'max_percentage'],
    price: '1500',
    useCase: 'Percentage-based spending limits'
  },
  vendor_trust: {
    file: 'curated/vendor_trust.onnx',
    name: 'Vendor Trust',
    description: 'Requires minimum vendor reputation score',
    category: 'Basic',
    inputs: ['vendor_trust', 'min_trust'],
    price: '1000',
    useCase: 'Marketplace vendor verification'
  },
  velocity_1h: {
    file: 'curated/velocity_1h.onnx',
    name: 'Hourly Velocity',
    description: 'Limits spending per hour',
    category: 'Velocity',
    inputs: ['amount', 'spent_1h', 'limit_1h'],
    price: '2000',
    useCase: 'Hourly rate limiting'
  },
  velocity_24h: {
    file: 'curated/velocity_24h.onnx',
    name: 'Daily Velocity',
    description: 'Limits spending per 24 hours',
    category: 'Velocity',
    inputs: ['amount', 'spent_24h', 'limit_24h'],
    price: '2000',
    useCase: 'Daily spending caps'
  },
  daily_limit: {
    file: 'curated/daily_limit.onnx',
    name: 'Daily Cap',
    description: 'Hard cap on daily spending',
    category: 'Velocity',
    inputs: ['amount', 'daily_spent', 'daily_cap'],
    price: '2000',
    useCase: 'Budget enforcement'
  },
  age_gate: {
    file: 'curated/age_gate.onnx',
    name: 'Age Gate',
    description: 'Checks minimum age requirement',
    category: 'Access',
    inputs: ['age', 'min_age'],
    price: '1000',
    useCase: 'Age-restricted purchases'
  },
  multi_factor: {
    file: 'curated/multi_factor.onnx',
    name: 'Multi-Factor',
    description: 'Combines balance + velocity + trust checks',
    category: 'Advanced',
    inputs: ['amount', 'balance', 'spent_24h', 'limit_24h', 'vendor_trust', 'min_trust'],
    price: '5000',
    useCase: 'High-security transactions'
  },
  composite_scoring: {
    file: 'curated/composite_scoring.onnx',
    name: 'Composite Scoring',
    description: 'Weighted risk score from multiple factors',
    category: 'Advanced',
    inputs: ['amount', 'balance', 'vendor_trust', 'user_history'],
    price: '4000',
    useCase: 'Advanced risk assessment'
  },
  risk_neural: {
    file: 'curated/risk_neural.onnx',
    name: 'Risk Neural',
    description: 'ML-based risk scoring with 47 operations',
    category: 'Advanced',
    inputs: ['amount', 'balance', 'velocity_1h', 'velocity_24h', 'vendor_trust'],
    price: '6000',
    useCase: 'Sophisticated fraud detection'
  }
};

/**
 * Parse X-PAYMENT header
 */
function parsePaymentHeader(headerValue) {
  try {
    const decoded = Buffer.from(headerValue, 'base64').toString('utf-8');
    return JSON.parse(decoded);
  } catch (error) {
    return null;
  }
}

/**
 * Encode X-PAYMENT-RESPONSE header
 */
function encodePaymentResponse(data) {
  const json = JSON.stringify(data);
  return Buffer.from(json).toString('base64');
}

/**
 * Generate x402 payment requirements for a model
 */
function generatePaymentRequirements(modelId, baseUrl) {
  const model = CURATED_MODELS[modelId];
  if (!model) return null;

  return {
    scheme: 'zkml-jolt',  // Custom scheme for zkML proofs
    network: 'jolt-atlas',  // Using JOLT Atlas prover
    maxAmountRequired: model.price,
    payTo: baseUrl + '/x402/verify-proof',
    asset: 'zkml-proof',  // Not a crypto asset, but a zkML proof
    resource: `/x402/authorize/${modelId}`,
    description: `${model.name}: ${model.description}`,
    mimeType: 'application/json',
    maxTimeoutSeconds: 30,
    extra: {
      modelId: modelId,
      modelName: model.name,
      category: model.category,
      inputs: model.inputs,
      useCase: model.useCase,
      proofType: 'jolt-atlas-onnx',
      onnxFile: model.file
    }
  };
}

/**
 * Verify zkML proof from X-PAYMENT header
 */
async function verifyZkmlProof(paymentData, modelId) {
  try {
    // Extract proof from payment payload
    const { payload } = paymentData;

    if (!payload || !payload.zkmlProof) {
      return { isValid: false, invalidReason: 'Missing zkML proof in payment' };
    }

    // Verify the proof matches the requested model
    if (payload.modelId !== modelId) {
      return {
        isValid: false,
        invalidReason: `Model mismatch: expected ${modelId}, got ${payload.modelId}`
      };
    }

    // Verify proof structure (actual cryptographic verification would happen here)
    const proof = payload.zkmlProof;
    if (!proof.approved !== undefined || !proof.output || !proof.verification) {
      return { isValid: false, invalidReason: 'Invalid proof structure' };
    }

    // TODO: Implement actual JOLT Atlas proof verification
    // For now, we trust the proof structure
    return {
      isValid: true,
      invalidReason: null,
      approved: proof.approved,
      output: proof.output
    };

  } catch (error) {
    return {
      isValid: false,
      invalidReason: `Proof verification error: ${error.message}`
    };
  }
}

/**
 * x402 middleware - checks for payment and enforces 402 responses
 */
function x402Middleware(options = {}) {
  const baseUrl = options.baseUrl || 'http://localhost:3001';

  return async (req, res, next) => {
    // Skip middleware for non-x402 endpoints
    if (!req.path.startsWith('/x402/authorize')) {
      return next();
    }

    // Extract model ID from path
    const modelMatch = req.path.match(/\/x402\/authorize\/(.+)/);
    if (!modelMatch) {
      return res.status(400).json({ error: 'Model ID required' });
    }

    const modelId = modelMatch[1];
    const model = CURATED_MODELS[modelId];

    if (!model) {
      return res.status(404).json({ error: 'Model not found' });
    }

    // Check for X-PAYMENT header
    const paymentHeader = req.headers['x-payment'];

    if (!paymentHeader) {
      // No payment provided - return 402 with payment requirements
      const paymentRequirements = generatePaymentRequirements(modelId, baseUrl);

      return res.status(402).json({
        x402Version: 1,
        accepts: [paymentRequirements],
        error: `Authorization proof required for ${model.name}`
      });
    }

    // Parse and verify payment
    const paymentData = parsePaymentHeader(paymentHeader);
    if (!paymentData) {
      return res.status(400).json({ error: 'Invalid X-PAYMENT header format' });
    }

    // Verify zkML proof
    const verification = await verifyZkmlProof(paymentData, modelId);

    if (!verification.isValid) {
      return res.status(402).json({
        x402Version: 1,
        accepts: [generatePaymentRequirements(modelId, baseUrl)],
        error: verification.invalidReason
      });
    }

    // Payment verified - attach to request and continue
    req.x402 = {
      verified: true,
      modelId: modelId,
      proof: paymentData.payload.zkmlProof,
      approved: verification.approved
    };

    next();
  };
}

module.exports = {
  x402Middleware,
  CURATED_MODELS,
  parsePaymentHeader,
  encodePaymentResponse,
  generatePaymentRequirements,
  verifyZkmlProof
};
