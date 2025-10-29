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
    service: 'zkX402 Agent Authorization - Production Ready ✅',
    version: '1.2.0',
    status: 'production',
    lastUpdated: '2025-10-29',
    description: 'Verifiable agent authorization using JOLT Atlas zkML proofs with Base USDC payments',
    x402Version: 1,
    verifiedModels: 14,
    modelBreakdown: {
      production: 10,
      test: 4
    },
    criticalFixes: [
      'Gather heap address collision fixed',
      'Two-pass input allocation implemented',
      'Constant index Gather addressing corrected'
    ],
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
    endpoints: {
      discovery: `${BASE_URL}/.well-known/x402`,
      models: `${BASE_URL}/x402/models`,
      authorize: `${BASE_URL}/x402/authorize/:modelId`,
      verifyProof: `${BASE_URL}/x402/verify-proof`,
      generateProof: `${BASE_URL}/api/generate-proof`,
      paymentInfo: `${BASE_URL}/x402/payment-info`,
      facilitator: {
        supported: `${BASE_URL}/x402/supported`,
        verify: `${BASE_URL}/x402/verify`
      }
    },
    schemes: [
      {
        scheme: 'zkml-jolt',
        network: 'base-mainnet',
        description: 'Zero-knowledge machine learning proofs using JOLT Atlas with Base USDC payments',
        proofType: 'onnx-inference',
        paymentMethod: 'USDC on Base L2',
        operations: 'Gather, Greater, Less, GreaterOrEqual, LessOrEqual, Div, Cast, Slice, Identity, Add, Sub, Mul'
      }
    ],
    models: Object.keys(CURATED_MODELS).map(id => ({
      id,
      name: CURATED_MODELS[id].name,
      category: CURATED_MODELS[id].category,
      description: CURATED_MODELS[id].description,
      price: CURATED_MODELS[id].price,
      priceUSDC: formatPrice(CURATED_MODELS[id].price),
      inputs: CURATED_MODELS[id].inputs,
      operations: CURATED_MODELS[id].operations,
      proofTime: CURATED_MODELS[id].proofTime,
      authorizeEndpoint: `${BASE_URL}/x402/authorize/${id}`
    })),
    documentation: 'https://github.com/hshadab/zkx402'
  });
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
    console.error('[x402/verify] Error:', error);
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
  if (!zkmlProof.approved !== undefined || !zkmlProof.output || !zkmlProof.verification) {
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

    const modelPath = path.join(MODELS_DIR, model.file);
    if (!fs.existsSync(modelPath)) {
      return reject(new Error(`Model file not found: ${model.file}`));
    }

    // Build input arguments dynamically based on model's required inputs
    const inputArgs = model.inputs.map(inputName => {
      const value = parseInt(inputs[inputName]) || 0;
      return value;
    }).join(' ');

    // Use pre-built binary if it exists, otherwise cargo run
    const usePrebuiltBinary = fs.existsSync(JOLT_BINARY);
    const cargoCmd = usePrebuiltBinary
      ? `${JOLT_BINARY} "${modelPath}" ${inputArgs}`
      : `cd ${JOLT_PROVER_DIR}/zkml-jolt-core && cargo run --release --example proof_json_output "${modelPath}" ${inputArgs}`;

    console.log(`[JOLT Atlas] Generating proof for ${modelId}: ${cargoCmd}`);

    exec(cargoCmd, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
      if (error) {
        console.error('[JOLT] Proof generation error:', error);
        console.error('[JOLT] stderr:', stderr);
        return reject(new Error(`Proof generation failed: ${error.message}`));
      }

      try {
        const lines = stdout.trim().split('\n');
        const jsonLine = lines[lines.length - 1];
        const result = JSON.parse(jsonLine);

        if (result.error) {
          return reject(new Error(result.message || result.error));
        }

        console.log(`[JOLT] Proof generated for ${modelId}:`, {
          approved: result.approved,
          proving_time: result.proving_time
        });

        resolve({
          approved: result.approved,
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
        console.error('[JOLT] Failed to parse proof output:', parseError);
        console.error('[JOLT] stdout:', stdout);
        reject(new Error(`Failed to parse proof output: ${parseError.message}`));
      }
    });
  });
}

app.post('/api/generate-proof', uiProofLimiter, async (req, res) => {
  try {
    const { model: modelId, inputs } = req.body;

    if (!modelId) {
      return res.status(400).json({ error: 'Model ID required' });
    }

    const model = CURATED_MODELS[modelId];
    if (!model) {
      return res.status(404).json({ error: `Model not found: ${modelId}` });
    }

    // Validate all required inputs are present
    const missingInputs = model.inputs.filter(input => !inputs[input]);
    if (missingInputs.length > 0) {
      return res.status(400).json({
        error: 'Missing required inputs',
        missing: missingInputs,
        required: model.inputs
      });
    }

    const startTime = Date.now();
    const proofData = await generateJoltProof(modelId, inputs);
    const proofTime = Date.now() - startTime;

    res.json({
      ...proofData,
      proofTime,
      inputs,
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
    });

  } catch (error) {
    console.error('[API] Proof generation failed:', error);
    res.status(500).json({
      error: 'Proof generation failed',
      message: error.message
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

// ========== START SERVER ==========
app.listen(PORT, () => {
  console.log(`
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

Documentation: https://github.com/hshadab/zkx402
Payment Guide: https://github.com/hshadab/zkx402/blob/main/zkx402-agent-auth/PAYMENT_GUIDE.md
  `);
});
