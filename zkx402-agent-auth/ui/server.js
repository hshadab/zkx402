const express = require('express');
const rateLimit = require('express-rate-limit');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const {
  x402Middleware,
  CURATED_MODELS,
  parsePaymentHeader,
  encodePaymentResponse,
  generatePaymentRequirements
} = require('./x402-middleware');

const { getPaymentInfo, formatPrice } = require('./base-payment');
const { registerAgentRoutes, setBaseUrl } = require('./agent-api-routes');
const webhookManager = require('./webhook-manager');
const analyticsManager = require('./analytics-manager');
const logger = require('./logger');
const cache = require('./cache');

// Gate blockchain monitor initialization behind env flag (default ON for compatibility)
const ENABLE_BLOCKCHAIN_MONITOR = process.env.ENABLE_BLOCKCHAIN_MONITOR === '0' ? false : true;
const blockchainMonitor = ENABLE_BLOCKCHAIN_MONITOR ? require('./blockchain-monitor') : null;

const app = express();
const PORT = process.env.PORT || 3001;
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;

// Paths
const MODELS_DIR = path.join(__dirname, '../policy-examples/onnx');
const JOLT_PROVER_DIR = path.join(__dirname, '../jolt-atlas-fork');  // Use fixed jolt-atlas-fork
const JOLT_BINARY = path.join(__dirname, '../jolt-atlas-fork/target/release/examples/proof_json_output');

// Multer config
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, MODELS_DIR),
  filename: (req, file, cb) => cb(null, file.originalname)
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    cb(null, file.originalname.endsWith('.onnx'));
  },
  limits: { fileSize: 100 * 1024 * 1024 }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(x402Middleware({ baseUrl: BASE_URL }));

// ========== RATE LIMITING ==========
// Free tier rate limiter for UI proof generation (5 proofs per day)
const uiProofLimiter = rateLimit({
  windowMs: 24 * 60 * 60 * 1000, // 24 hours
  max: 5, // 5 proofs per day per IP
  message: {
    error: 'Free tier limit exceeded',
    message: 'You have used your 5 free proofs for today. Try again tomorrow or integrate via x402 for unlimited access.',
    resetTime: '24 hours from first request',
    upgradeOptions: {
      x402Integration: {
        description: 'For production use, integrate via x402 protocol for pay-per-use unlimited access',
        endpoint: `${BASE_URL}/x402/authorize/:modelId`,
        documentation: 'https://github.com/hshadab/zkx402'
      },
      contact: {
        email: 'Upgrade inquiries: contact via GitHub Issues',
        github: 'https://github.com/hshadab/zkx402/issues'
      }
    }
  },
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
  // Skip rate limit for x402 authenticated requests
  skip: (req) => req.headers['x-payment'] !== undefined
});

// ========== x402 DISCOVERY ENDPOINT ==========
app.get('/.well-known/x402', (req, res) => {
  const paymentInfo = getPaymentInfo();

  res.json({
    service: 'zkX402 Privacy-Preserving Authorization for AI Agents',
    version: '1.3.0',
    status: 'production',
    lastUpdated: '2025-10-29',
    description: 'Privacy-preserving authorization using JOLT Atlas zkML proofs with Base USDC payments. Prove authorization without revealing sensitive data.',
    x402Version: 1,

    // Agent-friendly capabilities
    capabilities: {
      zkml_proofs: true,
      max_model_params: 1024,
      max_operations: 100,
      supported_input_types: ['int8', 'int16', 'int32', 'float32'],
      proof_time_range: '0.5s - 9s',
      supported_onnx_ops: ['Gather', 'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual', 'Div', 'Cast', 'Slice', 'Identity', 'Add', 'Sub', 'Mul', 'Clip', 'MatMul'],
      custom_model_upload: false,  // Future feature
      policy_composition: false     // Future feature
    },

    // Payment details
    pricing: {
      currency: 'USDC',
      network: 'base',
      chainId: paymentInfo.network.chainId,
      wallet: paymentInfo.paymentWallet,
      explorer: paymentInfo.network.explorer,
      tokenAddress: paymentInfo.token.address
    },

    // API endpoints for agents
    endpoints: {
      list_policies: `${BASE_URL}/api/policies`,
      get_policy_schema: `${BASE_URL}/api/policies/:id/schema`,
      generate_proof: `${BASE_URL}/api/generate-proof`,
      verify_proof: `${BASE_URL}/x402/verify-proof`,
      authorize: `${BASE_URL}/x402/authorize/:modelId`,
      discovery: `${BASE_URL}/.well-known/x402`,
      health: `${BASE_URL}/health`
    },

    // Pre-built policies (agent-readable)
    pre_built_policies: Object.keys(CURATED_MODELS).map(id => ({
      id,
      name: CURATED_MODELS[id].name,
      description: CURATED_MODELS[id].description,
      category: CURATED_MODELS[id].category,
      use_case: CURATED_MODELS[id].useCase,
      price_usdc: formatPrice(CURATED_MODELS[id].price),
      price_atomic: CURATED_MODELS[id].price,
      avg_proof_time: CURATED_MODELS[id].proofTime,
      operations: CURATED_MODELS[id].operations,
      complexity: CURATED_MODELS[id].operations <= 10 ? 'simple' : CURATED_MODELS[id].operations <= 30 ? 'medium' : 'advanced',
      inputs: CURATED_MODELS[id].inputs,
      endpoint: `${BASE_URL}/x402/authorize/${id}`,
      schema_url: `${BASE_URL}/api/policies/${id}/schema`
    })),

    // Backward compatibility
    verifiedModels: 14,
    modelBreakdown: {
      production: 10,
      test: 4
    },
    payment: {
      enabled: true,
      blockchain: paymentInfo.network.name,
      chainId: paymentInfo.network.chainId,
      token: paymentInfo.token.symbol,
      tokenAddress: paymentInfo.token.address,
      paymentWallet: paymentInfo.paymentWallet,
      explorer: paymentInfo.network.explorer,
      instructions: paymentInfo.instructions
    },

    documentation: 'https://github.com/hshadab/zkx402',
    sdk: {
      python: 'https://github.com/hshadab/zkx402-sdk-python',  // Future
      javascript: 'https://github.com/hshadab/zkx402-sdk-js'    // Future
    }
  });
});

// ========== COMPATIBILITY HEALTH ENDPOINT ==========
// Provide /api/health in addition to /health for older clients/tests
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    modelsAvailable: Object.keys(CURATED_MODELS).length
  });
});

// ========== COMPATIBILITY: MODELS LISTING ==========
// Legacy endpoint to list models (maps CURATED_MODELS)
app.get('/api/models', (req, res) => {
  try {
    const models = Object.entries(CURATED_MODELS).map(([id, model]) => {
      const filePath = path.join(MODELS_DIR, model.file);
      return {
        id,
        file: model.file,
        description: model.description,
        available: fs.existsSync(filePath)
      };
    });
    res.json({ models });
  } catch (error) {
    res.status(500).json({ error: 'Failed to list models' });
  }
});

// ========== COMPATIBILITY: VALIDATE MODELS ==========
app.get('/api/validate-models', (req, res) => {
  try {
    const models = Object.entries(CURATED_MODELS).map(([id, model]) => {
      const filePath = path.join(MODELS_DIR, model.file);
      return {
        id,
        file: model.file,
        exists: fs.existsSync(filePath),
        path: filePath
      };
    });
    const valid = models.every(m => m.exists);
    res.json({ valid, models });
  } catch (error) {
    res.status(500).json({ error: 'Failed to validate models' });
  }
});

// ========== x402 MODELS LISTING ==========
app.get('/x402/models', (req, res) => {
  const models = Object.entries(CURATED_MODELS).map(([id, model]) => ({
    id,
    name: model.name,
    description: model.description,
    category: model.category,
    useCase: model.useCase,
    inputs: model.inputs,
    price: model.price,
    priceUSDC: formatPrice(model.price),
    operations: model.operations,
    proofTime: model.proofTime,
    file: model.file,
    authorizeEndpoint: `${BASE_URL}/x402/authorize/${id}`,
    paymentRequirement: generatePaymentRequirements(id, BASE_URL)
  }));

  res.json({
    x402Version: 1,
    status: 'production',
    lastUpdated: '2025-10-29',
    models,
    totalModels: models.length,
    modelBreakdown: {
      production: models.filter(m => m.category !== 'Test').length,
      test: models.filter(m => m.category === 'Test').length
    }
  });
});

// ========== x402 PAYMENT INFO ENDPOINT ==========
app.get('/x402/payment-info', (req, res) => {
  res.json(getPaymentInfo());
});

// ========== x402 FACILITATOR ENDPOINTS ==========

// GET /x402/supported - List supported (scheme, network) pairs
app.get('/x402/supported', (req, res) => {
  res.json({
    x402Version: 1,
    supported: [
      {
        scheme: 'zkml-jolt',
        network: 'base-mainnet',
        description: 'Zero-knowledge ML proofs with Base USDC payments',
        features: [
          'On-chain payment verification',
          'JOLT Atlas zkML proof validation',
          'Direct settlement (no facilitator needed)'
        ],
        paymentToken: {
          symbol: 'USDC',
          address: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
          decimals: 6,
          network: 'Base Mainnet',
          chainId: 8453
        }
      }
    ]
  });
});

// POST /x402/verify - Verify payment payload without settling
app.post('/x402/verify', async (req, res) => {
  try {
    const { paymentHeader, paymentRequirements } = req.body;

    if (!paymentHeader) {
      return res.status(400).json({
        isValid: false,
        invalidReason: 'Missing paymentHeader'
      });
    }

    if (!paymentRequirements) {
      return res.status(400).json({
        isValid: false,
        invalidReason: 'Missing paymentRequirements'
      });
    }

    // Parse payment header
    const paymentData = parsePaymentHeader(paymentHeader);
    if (!paymentData) {
      return res.status(400).json({
        isValid: false,
        invalidReason: 'Invalid payment header format'
      });
    }

    // Extract model ID from payment requirements
    const modelId = paymentRequirements.extra?.modelId || paymentRequirements.resource?.split('/').pop();
    if (!modelId) {
      return res.status(400).json({
        isValid: false,
        invalidReason: 'Cannot determine model ID from payment requirements'
      });
    }

    // Verify the payment
    const { verifyZkmlProof } = require('./x402-middleware');
    const verification = await verifyZkmlProof(paymentData, modelId);

    if (!verification.isValid) {
      return res.json({
        isValid: false,
        invalidReason: verification.invalidReason,
        paymentDetails: verification.paymentDetails
      });
    }

    // Payment verified successfully
    res.json({
      isValid: true,
      invalidReason: null,
      details: {
        modelId,
        approved: verification.approved,
        paymentVerified: true,
        proofVerified: true,
        paymentDetails: verification.paymentDetails
      }
    });

  } catch (error) {
    logger.error('[x402/verify] Error', { error: error.message, stack: error.stack });
    res.status(500).json({
      isValid: false,
      invalidReason: `Verification error: ${error.message}`
    });
  }
});

// ========== x402 AUTHORIZATION ENDPOINT ==========
app.post('/x402/authorize/:modelId', async (req, res) => {
  // x402 middleware has already verified payment
  if (!req.x402 || !req.x402.verified) {
    return res.status(500).json({ error: 'x402 middleware verification failed' });
  }

  const { modelId, proof, approved } = req.x402;

  // Return authorization result with X-PAYMENT-RESPONSE header
  const paymentResponse = encodePaymentResponse({
    x402Version: 1,
    modelId,
    approved,
    timestamp: new Date().toISOString(),
    proofVerified: true,
    output: proof.output
  });

  res.setHeader('X-PAYMENT-RESPONSE', paymentResponse);
  res.json({
    authorized: approved,
    modelId,
    modelName: CURATED_MODELS[modelId].name,
    proof: {
      verified: true,
      approved,
      output: proof.output
    },
    timestamp: new Date().toISOString()
  });
});

// ========== x402 PROOF VERIFICATION ENDPOINT ==========
app.post('/x402/verify-proof', async (req, res) => {
  const { modelId, zkmlProof, inputs } = req.body;

  if (!modelId || !zkmlProof) {
    return res.status(400).json({
      isValid: false,
      invalidReason: 'Missing modelId or zkmlProof'
    });
  }

  const model = CURATED_MODELS[modelId];
  if (!model) {
    return res.status(404).json({
      isValid: false,
      invalidReason: 'Model not found'
    });
  }

  // Verify proof structure
  if (zkmlProof.approved === undefined || !zkmlProof.output || !zkmlProof.verification) {
    return res.status(400).json({
      isValid: false,
      invalidReason: 'Invalid proof structure'
    });
  }

  // TODO: Implement cryptographic verification of JOLT Atlas proof
  // For now, we trust the structure
  res.json({
    isValid: true,
    approved: zkmlProof.approved,
    modelId,
    timestamp: new Date().toISOString()
  });
});

// ========== PROOF GENERATION (Updated for all models) ==========
function generateJoltProof(modelId, inputs) {
  return new Promise((resolve, reject) => {
    const model = CURATED_MODELS[modelId];
    if (!model) {
      return reject(new Error(`Model not found: ${modelId}`));
    }

    // Resolve model path with optional division-free fallback
    let modelRelPath = model.file;
    const preferNoDiv = process.env.PREFER_NO_DIV === '1';
    if (preferNoDiv && modelRelPath.endsWith('.onnx')) {
      const noDivCandidate = modelRelPath.replace(/\.onnx$/i, '_no_div.onnx');
      const noDivFullPath = path.join(MODELS_DIR, noDivCandidate);
      if (fs.existsSync(noDivFullPath)) {
        modelRelPath = noDivCandidate;
        logger.info('Using division-free model variant', { modelId, variant: noDivCandidate });
      }
    }

    const modelPath = path.join(MODELS_DIR, modelRelPath);
    if (!fs.existsSync(modelPath)) {
      return reject(new Error(`Model file not found: ${modelRelPath}`));
    }

    // Build input arguments dynamically based on model's required inputs
    const inputArgs = model.inputs.map(inputName => {
      const raw = inputs ? inputs[inputName] : undefined;
      const value = raw === undefined || raw === null || raw === '' ? 0 : parseInt(raw);
      return Number.isNaN(value) ? 0 : value;
    }).join(' ');

    // Use pre-built binary if it exists, otherwise cargo run
    const usePrebuiltBinary = fs.existsSync(JOLT_BINARY);
    // Enable prover tuning flags via environment passthrough
    const proverEnv = {
      JOLT_TRACE_TRANSCRIPT: process.env.JOLT_TRACE_TRANSCRIPT || undefined,
      JOLT_TRACE_DIV: process.env.JOLT_TRACE_DIV || undefined,
      JOLT_REWRITE_CONST_DIV: process.env.JOLT_REWRITE_CONST_DIV || undefined,
      JOLT_DIV_V2: process.env.JOLT_DIV_V2 || undefined,
      JOLT_SUMCHECK_CHUNK: process.env.JOLT_SUMCHECK_CHUNK || undefined,
      JOLT_SUMCHECK_CHUNK_SIZE: process.env.JOLT_SUMCHECK_CHUNK_SIZE || undefined,
      JOLT_SUMCHECK_BIND_LOW2HIGH: process.env.JOLT_SUMCHECK_BIND_LOW2HIGH || undefined,
    };
    const envPrefix = Object.entries(proverEnv)
      .filter(([_, v]) => v !== undefined)
      .map(([k, v]) => `${k}=${v}`)
      .join(' ');
    const cargoCmd = usePrebuiltBinary
      ? `${envPrefix} ${JOLT_BINARY} "${modelPath}" ${inputArgs}`.trim()
      : `${envPrefix} cd ${JOLT_PROVER_DIR}/zkml-jolt-core && ${envPrefix} cargo run --release --example proof_json_output "${modelPath}" ${inputArgs}`.trim();

    logger.info('Generating JOLT proof', { modelId, command: cargoCmd.substring(0, 150) });

    exec(cargoCmd, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
      if (error) {
        logger.error('JOLT proof generation error', {
          modelId,
          error: error.message,
          stderr: stderr.substring(0, 500)
        });
        return reject(new Error(`Proof generation failed: ${error.message}`));
      }

      try {
        const lines = stdout.trim().split('\n');
        const jsonLine = lines[lines.length - 1];
        const result = JSON.parse(jsonLine);

        if (result.error) {
          return reject(new Error(result.message || result.error));
        }

        // Derive approval status from output value
        // For authorization models: output=1 means approved, output=0 means denied
        const isApproved = result.output === 1;

        logger.info('JOLT proof generated successfully', {
          modelId,
          output: result.output,
          approved: isApproved,
          proving_time: result.proving_time
        });

        resolve({
          approved: isApproved,
          output: result.output,
          verification: result.verification,
          proofSize: result.proof_size,
          verificationTime: result.verification_time,
          operations: result.operations,
          zkmlProof: result.zkml_proof,
          modelId,
          modelName: model.name
        });
      } catch (parseError) {
        logger.error('Failed to parse JOLT proof output', {
          modelId,
          error: parseError.message,
          stdout: stdout.substring(0, 500)
        });
        reject(new Error(`Failed to parse proof output: ${parseError.message}`));
      }
    });
  });
}

app.post('/api/generate-proof', uiProofLimiter, async (req, res) => {
  try {
    const { model: modelId, inputs, webhook_id } = req.body;

    if (!modelId) {
      logger.warn('Proof generation request missing model ID', { ip: req.ip });
      return res.status(400).json({ error: 'Model ID required' });
    }

    const model = CURATED_MODELS[modelId];
    if (!model) {
      logger.warn('Proof generation request for unknown model', { modelId, ip: req.ip });
      return res.status(404).json({ error: `Model not found: ${modelId}` });
    }

    // Validate all required inputs are present (treat 0 as valid)
    const inputObj = inputs || {};
    const missingInputs = model.inputs.filter(input => !Object.prototype.hasOwnProperty.call(inputObj, input));
    if (missingInputs.length > 0) {
      logger.warn('Proof generation request with missing inputs', {
        modelId,
        missing: missingInputs,
        ip: req.ip
      });
      return res.status(400).json({
        error: 'Missing required inputs',
        missing: missingInputs,
        required: model.inputs
      });
    }

    // If webhook_id provided, validate it exists
    if (webhook_id) {
      const webhook = webhookManager.getWebhook(webhook_id);
      if (!webhook) {
        return res.status(400).json({
          error: 'Webhook not found',
          webhook_id
        });
      }
    }

    // Create proof request for webhook tracking (if webhook_id provided)
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    if (webhook_id) {
      webhookManager.createProofRequest(requestId, webhook_id, modelId, inputs);
    }

    // Check cache first
    const cachedProof = await cache.getProofFromCache(modelId, inputs);
    if (cachedProof) {
      logger.info('Returning cached proof', { modelId, requestId, ip: req.ip });

      const result = {
        ...cachedProof,
        request_id: requestId,
        cached: true,
        x402Payment: {
          header: encodePaymentResponse({
            x402Version: 1,
            scheme: 'zkml-jolt',
            network: 'jolt-atlas',
            payload: {
              modelId,
              zkmlProof: cachedProof
            }
          })
        }
      };

      // Trigger webhook if configured
      if (webhook_id) {
        await webhookManager.completeProofRequest(requestId, result);
      }

      // Log to analytics
      analyticsManager.logRequest({
        endpoint: '/api/generate-proof',
        method: 'POST',
        modelId,
        success: true,
        responseTime: 0,
        cached: true,
        userAgent: req.headers['user-agent'],
        ip: req.ip,
        hasPaidHeader: !!req.headers['x402-paid']
      });

      return res.json(result);
    }

    logger.info('Generating new proof', { modelId, requestId, ip: req.ip });

    const startTime = Date.now();
    const proofData = await generateJoltProof(modelId, inputs);
    const proofTime = Date.now() - startTime;

    // Store proof in cache
    await cache.storeProofInCache(modelId, inputs, proofData);

    const result = {
      ...proofData,
      proofTime,
      inputs,
      request_id: requestId,
      cached: false,
      x402Payment: {
        header: encodePaymentResponse({
          x402Version: 1,
          scheme: 'zkml-jolt',
          network: 'jolt-atlas',
          payload: {
            modelId,
            zkmlProof: proofData
          }
        })
      }
    };

    // Trigger webhook if configured
    if (webhook_id) {
      await webhookManager.completeProofRequest(requestId, result);
    }

    // Log successful request to analytics
    analyticsManager.logRequest({
      endpoint: '/api/generate-proof',
      method: 'POST',
      modelId,
      success: true,
      responseTime: proofTime,
      cached: false,
      userAgent: req.headers['user-agent'],
      ip: req.ip,
      hasPaidHeader: !!req.headers['x402-paid']
    });

    logger.info('Proof generation completed', {
      modelId,
      requestId,
      proofTime: `${proofTime}ms`,
      approved: proofData.approved
    });

    res.json(result);

  } catch (error) {
    logger.error('Proof generation failed', {
      modelId: req.body.model,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined,
      ip: req.ip
    });

    // Log failed request to analytics
    analyticsManager.logRequest({
      endpoint: '/api/generate-proof',
      method: 'POST',
      modelId: req.body.model,
      success: false,
      errorMessage: error.message,
      userAgent: req.headers['user-agent'],
      ip: req.ip,
      hasPaidHeader: !!req.headers['x402-paid']
    });

    res.status(500).json({
      error: 'Proof generation failed',
      message: error.message,
      // Hide stack trace in production
      ...(process.env.NODE_ENV === 'development' && { stack: error.stack })
    });
  }
});

// ========== MODEL UPLOAD ==========
app.post('/api/upload-model', upload.single('model'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  res.json({
    success: true,
    filename: req.file.filename,
    path: req.file.path,
    size: req.file.size
  });
});

// ========== HEALTH CHECK ==========
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'zkX402',
    version: '1.0.0',
    x402Enabled: true,
    models: Object.keys(CURATED_MODELS).length,
    timestamp: new Date().toISOString()
  });
});

// ========== ANALYTICS API ENDPOINTS ==========
app.get('/api/analytics/stats', (req, res) => {
  try {
    const stats = analyticsManager.getStats();
    res.json(stats);
  } catch (error) {
    logger.error('Error getting analytics stats', { error: error.message });
    res.status(500).json({ error: 'Failed to get analytics stats' });
  }
});

app.get('/api/analytics/models', (req, res) => {
  try {
    const breakdown = analyticsManager.getModelBreakdown();
    res.json(breakdown);
  } catch (error) {
    logger.error('Error getting model breakdown', { error: error.message });
    res.status(500).json({ error: 'Failed to get model breakdown' });
  }
});

app.get('/api/analytics/timeseries', (req, res) => {
  try {
    const hours = parseInt(req.query.hours) || 24;
    const timeseries = analyticsManager.getTimeSeries(hours);
    res.json(timeseries);
  } catch (error) {
    logger.error('Error getting timeseries', { error: error.message });
    res.status(500).json({ error: 'Failed to get timeseries data' });
  }
});

// Get blockchain payment stats
app.get('/api/analytics/blockchain', async (req, res) => {
  try {
    if (!ENABLE_BLOCKCHAIN_MONITOR || !blockchainMonitor) {
      return res.json({ enabled: false, message: 'Blockchain monitor disabled' });
    }
    const stats = await blockchainMonitor.getStats();
    res.json({ enabled: true, ...stats });
  } catch (error) {
    logger.error('Error getting blockchain stats', { error: error.message });
    res.status(500).json({ error: 'Failed to get blockchain stats' });
  }
});

// ========== AGENT API ROUTES ==========
// Register agent-friendly API endpoints
// IMPORTANT: Must come BEFORE catch-all routes
setBaseUrl(BASE_URL);
registerAgentRoutes(app);

// ========== SERVE REACT UI (PRODUCTION) ==========
// Serve static files from Vite build output
const distPath = path.join(__dirname, 'dist');
if (fs.existsSync(distPath)) {
  logger.info('Serving React UI from dist directory');
  app.use(express.static(distPath));

  // Serve index.html for all other routes (SPA fallback)
  app.get('*', (req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
  });
} else {
  logger.warn('No dist directory found - UI not built yet');
  app.get('*', (req, res) => {
    res.status(503).json({
      error: 'UI not available',
      message: 'React UI has not been built. Run "npm run build" to build the UI.',
      api: {
        health: '/health',
        discovery: '/.well-known/x402',
        models: '/x402/models'
      }
    });
  });
}

// ========== AGENT DISCOVERY ENDPOINTS ==========

// Serve OpenAPI specification
app.get('/openapi.yaml', (req, res) => {
  res.sendFile(path.join(__dirname, 'openapi.yaml'));
});

app.get('/openapi.json', (req, res) => {
  // Convert YAML to JSON on the fly
  const yaml = require('js-yaml');
  const openapi = yaml.load(fs.readFileSync(path.join(__dirname, 'openapi.yaml'), 'utf8'));
  res.json(openapi);
});

// Serve AgentCard for AP2 discoverability
app.get('/.well-known/agentcard', (req, res) => {
  res.sendFile(path.join(__dirname, 'agentcard.json'));
});

app.get('/agentcard.json', (req, res) => {
  res.sendFile(path.join(__dirname, 'agentcard.json'));
});

// Add cache stats endpoint
app.get('/api/cache/stats', (req, res) => {
  try {
    const stats = cache.getCacheStats();
    res.json(stats);
  } catch (error) {
    logger.error('Error getting cache stats', { error: error.message });
    res.status(500).json({ error: 'Failed to get cache stats' });
  }
});

// Clear cache endpoint (admin)
app.post('/api/cache/clear', async (req, res) => {
  try {
    const { modelId } = req.body;
    let deletedCount;

    if (modelId) {
      deletedCount = await cache.clearModelCache(modelId);
      logger.info('Cache cleared for model', { modelId, deletedCount });
    } else {
      deletedCount = await cache.clearAllCache();
      logger.info('All cache cleared', { deletedCount });
    }

    res.json({
      success: true,
      deletedCount,
      modelId: modelId || 'all'
    });
  } catch (error) {
    logger.error('Error clearing cache', { error: error.message });
    res.status(500).json({ error: 'Failed to clear cache' });
  }
});

// ========== START SERVER ==========
async function startServer() {
  // Initialize Redis cache
  await cache.initializeRedis();

  app.listen(PORT, () => {
    logger.info(`
╔═══════════════════════════════════════════════════════════╗
║                   zkX402 API Server                       ║
║          Verifiable Agent Authorization (x402)            ║
╠═══════════════════════════════════════════════════════════╣
║  Port:           ${PORT.toString().padEnd(42)}║
║  x402 Discovery: /.well-known/x402                       ║
║  Models:         ${Object.keys(CURATED_MODELS).length.toString().padEnd(42)}║
║  Status:         🟢 READY                                 ║
╚═══════════════════════════════════════════════════════════╝

x402 Endpoints:
  GET  /.well-known/x402        - Service discovery
  GET  /x402/models             - List authorization models
  POST /x402/authorize/:modelId - Authorize with proof (402 flow)
  POST /x402/verify-proof       - Verify zkML proof
  GET  /x402/supported          - Supported schemes/networks
  POST /x402/verify             - Verify payment payload

Standard Endpoints:
  POST /api/generate-proof      - Generate zkML proof
  POST /api/upload-model        - Upload custom model
  GET  /health                  - Health check
  GET  /api/cache/stats         - Cache statistics

Documentation: https://github.com/hshadab/zkx402
Payment Guide: https://github.com/hshadab/zkx402/blob/main/zkx402-agent-auth/PAYMENT_GUIDE.md
    `);

    logger.info('zkX402 server started', {
      port: PORT,
      environment: process.env.NODE_ENV || 'development',
      cacheEnabled: cache.getCacheStats().enabled,
      models: Object.keys(CURATED_MODELS).length
    });
  });
}

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  await cache.closeRedis();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  await cache.closeRedis();
  process.exit(0);
});

startServer().catch((error) => {
  logger.error('Failed to start server', { error: error.message, stack: error.stack });
  process.exit(1);
});
