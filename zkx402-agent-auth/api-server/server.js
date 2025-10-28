/**
 * zkX402 External API Server
 * Production-ready REST API for JOLT Atlas proof generation
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const winston = require('winston');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 4000;

// Configure logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
  methods: ['GET', 'POST'],
  credentials: true
}));
app.use(express.json());

// Rate limiting
const proofLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: process.env.RATE_LIMIT || 100,
  message: { error: 'Too many requests', message: 'Please try again later' },
  standardHeaders: true,
  legacyHeaders: false,
});

// Request logging middleware
app.use((req, res, next) => {
  const requestId = uuidv4();
  req.requestId = requestId;

  logger.info('Incoming request', {
    requestId,
    method: req.method,
    path: req.path,
    ip: req.ip,
    userAgent: req.get('user-agent')
  });

  next();
});

// Paths
const MODELS_DIR = path.join(__dirname, '../policy-examples/onnx');
const JOLT_PROVER_DIR = path.join(__dirname, '../jolt-prover');

// Model configurations
const MODEL_CONFIGS = {
  simple_auth: {
    file: 'simple_auth.onnx',
    description: 'Simple rule-based authorization',
    inputCount: 5,
    estimatedTime: '700ms'
  },
  neural_auth: {
    file: 'neural_auth.onnx',
    description: 'Neural network authorization',
    inputCount: 5,
    estimatedTime: '1.5s'
  },
  comparison_demo: {
    file: 'comparison_demo.onnx',
    description: 'Comparison operations demo',
    inputCount: 3,
    estimatedTime: '300ms'
  },
  tensor_ops_demo: {
    file: 'tensor_ops_demo.onnx',
    description: 'Tensor operations demo',
    inputCount: 1,
    estimatedTime: '300ms'
  },
  matmult_1d_demo: {
    file: 'matmult_1d_demo.onnx',
    description: 'MatMult 1D output demo',
    inputCount: 1,
    estimatedTime: '400ms'
  }
};

/**
 * Generate JOLT Atlas proof
 */
async function generateProof(model, inputs, requestId) {
  return new Promise((resolve, reject) => {
    const modelPath = path.join(MODELS_DIR, MODEL_CONFIGS[model].file);

    if (!fs.existsSync(modelPath)) {
      return reject(new Error(`Model not found: ${MODEL_CONFIGS[model].file}`));
    }

    const { amount, balance, velocity_1h, velocity_24h, vendor_trust } = inputs;

    const cargoCmd = `cd ${JOLT_PROVER_DIR} && cargo run --release --example proof_json_output "${modelPath}" ${amount} ${balance} ${velocity_1h} ${velocity_24h} ${vendor_trust}`;

    logger.info('Generating proof', { requestId, model, inputs });

    const startTime = Date.now();

    exec(cargoCmd, { maxBuffer: 1024 * 1024 * 10, timeout: 120000 }, (error, stdout, stderr) => {
      const duration = Date.now() - startTime;

      if (error) {
        logger.error('Proof generation failed', {
          requestId,
          model,
          error: error.message,
          stderr,
          duration
        });
        return reject(new Error(`Proof generation failed: ${error.message}`));
      }

      try {
        const lines = stdout.trim().split('\n');
        const jsonLine = lines[lines.length - 1];
        const result = JSON.parse(jsonLine);

        if (result.error) {
          logger.error('Proof error', { requestId, error: result });
          return reject(new Error(result.message || result.error));
        }

        logger.info('Proof generated successfully', {
          requestId,
          model,
          approved: result.approved,
          duration: `${duration}ms`,
          proofSize: result.proof_size
        });

        resolve({
          requestId,
          approved: result.approved,
          output: result.output,
          verification: result.verification,
          proofSize: result.proof_size,
          provingTime: result.proving_time,
          verificationTime: result.verification_time,
          operations: result.operations,
          zkmlProof: result.zkml_proof,
          metadata: {
            model,
            inputs,
            timestamp: new Date().toISOString(),
            duration: `${duration}ms`
          }
        });
      } catch (parseError) {
        logger.error('Failed to parse proof output', {
          requestId,
          error: parseError.message,
          stdout
        });
        reject(new Error(`Failed to parse proof output: ${parseError.message}`));
      }
    });
  });
}

// Routes

/**
 * GET /api/v1/health
 * Health check
 */
app.get('/api/v1/health', (req, res) => {
  res.json({
    status: 'healthy',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    modelsAvailable: Object.keys(MODEL_CONFIGS).length
  });
});

/**
 * GET /api/v1/models
 * List available models
 */
app.get('/api/v1/models', (req, res) => {
  const models = Object.entries(MODEL_CONFIGS).map(([id, config]) => ({
    id,
    ...config,
    available: fs.existsSync(path.join(MODELS_DIR, config.file))
  }));

  res.json({
    models,
    count: models.length
  });
});

/**
 * POST /api/v1/proof
 * Generate authorization proof
 */
app.post(
  '/api/v1/proof',
  proofLimiter,
  [
    body('model').isString().isIn(Object.keys(MODEL_CONFIGS)),
    body('inputs').isObject(),
    body('inputs.amount').isString().matches(/^\d+$/),
    body('inputs.balance').isString().matches(/^\d+$/),
    body('inputs.velocity_1h').isString().matches(/^\d+$/),
    body('inputs.velocity_24h').isString().matches(/^\d+$/),
    body('inputs.vendor_trust').isString().matches(/^\d+$/),
  ],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      logger.warn('Validation failed', {
        requestId: req.requestId,
        errors: errors.array()
      });
      return res.status(400).json({
        error: 'Validation failed',
        message: 'Invalid request parameters',
        details: errors.array()
      });
    }

    const { model, inputs } = req.body;

    try {
      const result = await generateProof(model, inputs, req.requestId);
      res.json(result);
    } catch (error) {
      logger.error('Proof request failed', {
        requestId: req.requestId,
        error: error.message
      });

      res.status(500).json({
        error: 'Proof generation failed',
        message: error.message,
        requestId: req.requestId
      });
    }
  }
);

/**
 * POST /api/v1/proof/batch
 * Generate multiple proofs in parallel
 */
app.post(
  '/api/v1/proof/batch',
  proofLimiter,
  [
    body('requests').isArray().isLength({ min: 1, max: 10 }),
    body('requests.*.model').isString().isIn(Object.keys(MODEL_CONFIGS)),
    body('requests.*.inputs').isObject(),
  ],
  async (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        error: 'Validation failed',
        details: errors.array()
      });
    }

    const { requests } = req.body;

    try {
      const promises = requests.map((request, index) =>
        generateProof(request.model, request.inputs, `${req.requestId}-${index}`)
          .then(result => ({ success: true, result }))
          .catch(error => ({ success: false, error: error.message }))
      );

      const results = await Promise.all(promises);

      const successful = results.filter(r => r.success).length;
      const failed = results.length - successful;

      logger.info('Batch proof completed', {
        requestId: req.requestId,
        total: results.length,
        successful,
        failed
      });

      res.json({
        requestId: req.requestId,
        total: results.length,
        successful,
        failed,
        results
      });
    } catch (error) {
      logger.error('Batch proof failed', {
        requestId: req.requestId,
        error: error.message
      });

      res.status(500).json({
        error: 'Batch proof failed',
        message: error.message,
        requestId: req.requestId
      });
    }
  }
);

// Error handler
app.use((err, req, res, next) => {
  logger.error('Unhandled error', {
    requestId: req.requestId,
    error: err.message,
    stack: err.stack
  });

  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'production' ? 'An error occurred' : err.message,
    requestId: req.requestId
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`,
    availableRoutes: [
      'GET /api/v1/health',
      'GET /api/v1/models',
      'POST /api/v1/proof',
      'POST /api/v1/proof/batch'
    ]
  });
});

// Start server
app.listen(PORT, () => {
  logger.info('zkX402 API Server started', {
    port: PORT,
    environment: process.env.NODE_ENV || 'development',
    modelsDir: MODELS_DIR,
    joltProverDir: JOLT_PROVER_DIR
  });

  console.log('='.repeat(60));
  console.log('zkX402 External API Server');
  console.log('='.repeat(60));
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET  /api/v1/health           - Health check`);
  console.log(`  GET  /api/v1/models           - List models`);
  console.log(`  POST /api/v1/proof            - Generate proof`);
  console.log(`  POST /api/v1/proof/batch      - Batch proof generation`);
  console.log('='.repeat(60));
});

module.exports = app;
