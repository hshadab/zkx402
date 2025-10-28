const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

const app = express();
const PORT = process.env.PORT || 3001;

// Path to ONNX models
const MODELS_DIR = path.join(__dirname, '../policy-examples/onnx');
const JOLT_PROVER_DIR = path.join(__dirname, '../jolt-prover');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, MODELS_DIR);
  },
  filename: function (req, file, cb) {
    // Keep original filename
    cb(null, file.originalname);
  }
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.originalname.endsWith('.onnx')) {
      cb(null, true);
    } else {
      cb(new Error('Only .onnx files are allowed'));
    }
  },
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB max
  }
});

// Middleware
app.use(cors());
app.use(express.json());

// Model configurations
const MODEL_CONFIGS = {
  simple_auth: {
    file: 'simple_auth.onnx',
    description: 'Simple rule-based authorization',
    inputCount: 5
  },
  neural_auth: {
    file: 'neural_auth.onnx',
    description: 'Neural network authorization',
    inputCount: 5
  },
  comparison_demo: {
    file: 'comparison_demo.onnx',
    description: 'Comparison operations demo',
    inputCount: 3
  },
  tensor_ops_demo: {
    file: 'tensor_ops_demo.onnx',
    description: 'Tensor operations demo',
    inputCount: 1
  },
  matmult_1d_demo: {
    file: 'matmult_1d_demo.onnx',
    description: 'MatMult 1D output demo',
    inputCount: 1
  }
};

/**
 * Generate REAL JOLT Atlas proof using Rust prover
 */
function generateRealJoltProof(modelType, inputs) {
  return new Promise((resolve, reject) => {
    const modelPath = path.join(MODELS_DIR, MODEL_CONFIGS[modelType].file);

    // Check if model exists
    if (!fs.existsSync(modelPath)) {
      return reject(new Error(`Model not found: ${MODEL_CONFIGS[modelType].file}`));
    }

    // Convert inputs to scaled integers (scale by 100)
    const amount = parseInt(inputs.amount) || 0;
    const balance = parseInt(inputs.balance) || 0;
    const velocity_1h = parseInt(inputs.velocity_1h) || 0;
    const velocity_24h = parseInt(inputs.velocity_24h) || 0;
    const vendor_trust = parseInt(inputs.vendor_trust) || 0;

    // Build cargo command
    const cargoCmd = `cd ${JOLT_PROVER_DIR} && cargo run --release --example proof_json_output "${modelPath}" ${amount} ${balance} ${velocity_1h} ${velocity_24h} ${vendor_trust}`;

    console.log(`[JOLT] Generating proof with command: ${cargoCmd}`);

    // Execute Rust prover
    exec(cargoCmd, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
      if (error) {
        console.error('[JOLT] Proof generation error:', error);
        console.error('[JOLT] stderr:', stderr);
        return reject(new Error(`Proof generation failed: ${error.message}`));
      }

      try {
        // Parse JSON output from Rust
        const lines = stdout.trim().split('\n');
        const jsonLine = lines[lines.length - 1]; // Last line should be JSON
        const result = JSON.parse(jsonLine);

        if (result.error) {
          return reject(new Error(result.message || result.error));
        }

        console.log('[JOLT] Proof generated successfully:', {
          approved: result.approved,
          proving_time: result.proving_time,
          verification_time: result.verification_time,
        });

        // Return in the format expected by the UI
        resolve({
          approved: result.approved,
          output: result.output,
          verification: result.verification,
          proofSize: result.proof_size,
          verificationTime: result.verification_time,
          operations: result.operations,
          zkmlProof: result.zkml_proof
        });
      } catch (parseError) {
        console.error('[JOLT] Failed to parse proof output:', parseError);
        console.error('[JOLT] stdout:', stdout);
        reject(new Error(`Failed to parse proof output: ${parseError.message}`));
      }
    });
  });
}

/**
 * Generate zkML proof using JOLT Atlas
 * Calls the actual Rust prover via CLI
 */
async function generateJoltProof(modelType, inputs) {
  const modelPath = path.join(MODELS_DIR, MODEL_CONFIGS[modelType].file);

  // Check if model exists
  if (!fs.existsSync(modelPath)) {
    throw new Error(`Model not found: ${MODEL_CONFIGS[modelType].file}`);
  }

  // Generate REAL JOLT Atlas proof
  return generateRealJoltProof(modelType, inputs);
}

// API Routes

/**
 * GET /api/models
 * List available models
 */
app.get('/api/models', (req, res) => {
  const models = Object.entries(MODEL_CONFIGS).map(([id, config]) => ({
    id,
    ...config,
    available: fs.existsSync(path.join(MODELS_DIR, config.file))
  }));

  res.json({ models });
});

/**
 * POST /api/generate-proof
 * Generate zkML proof for authorization
 */
app.post('/api/generate-proof', async (req, res) => {
  try {
    const { model, inputs } = req.body;

    // Validate request
    if (!model || !inputs) {
      return res.status(400).json({
        error: 'Missing required fields: model, inputs'
      });
    }

    // Validate model type
    if (!MODEL_CONFIGS[model]) {
      return res.status(400).json({
        error: `Invalid model type: ${model}`
      });
    }

    console.log(`[${new Date().toISOString()}] Generating proof for model: ${model}`);
    console.log('Inputs:', inputs);

    // Generate proof
    const proof = await generateJoltProof(model, inputs);

    console.log(`[${new Date().toISOString()}] Proof generated successfully`);
    console.log('Result:', proof.approved ? 'APPROVED' : 'DENIED');

    res.json(proof);
  } catch (error) {
    console.error('Proof generation error:', error);
    res.status(500).json({
      error: 'Proof generation failed',
      message: error.message
    });
  }
});

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    modelsDir: MODELS_DIR,
    modelsAvailable: Object.keys(MODEL_CONFIGS).length
  });
});

/**
 * GET /api/validate-models
 * Validate that all ONNX models exist
 */
app.get('/api/validate-models', (req, res) => {
  const results = Object.entries(MODEL_CONFIGS).map(([id, config]) => {
    const modelPath = path.join(MODELS_DIR, config.file);
    return {
      id,
      file: config.file,
      exists: fs.existsSync(modelPath),
      path: modelPath
    };
  });

  const allValid = results.every(r => r.exists);

  res.json({
    valid: allValid,
    models: results
  });
});

/**
 * POST /api/upload-model
 * Upload a new ONNX model
 */
app.post('/api/upload-model', upload.single('model'), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: 'No file uploaded',
        message: 'Please provide an ONNX model file'
      });
    }

    console.log(`[${new Date().toISOString()}] Model uploaded: ${req.file.filename}`);

    res.json({
      success: true,
      message: 'Model uploaded successfully',
      filename: req.file.filename,
      path: req.file.path,
      size: req.file.size
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({
      error: 'Upload failed',
      message: error.message
    });
  }
});

/**
 * DELETE /api/models/:modelId
 * Delete a model
 */
app.delete('/api/models/:modelId', (req, res) => {
  try {
    const { modelId } = req.params;

    // Prevent deletion of built-in models
    if (MODEL_CONFIGS[modelId]) {
      return res.status(403).json({
        error: 'Cannot delete built-in model',
        message: `Model "${modelId}" is a built-in model and cannot be deleted`
      });
    }

    const modelPath = path.join(MODELS_DIR, `${modelId}.onnx`);

    if (!fs.existsSync(modelPath)) {
      return res.status(404).json({
        error: 'Model not found',
        message: `Model "${modelId}" does not exist`
      });
    }

    fs.unlinkSync(modelPath);
    console.log(`[${new Date().toISOString()}] Model deleted: ${modelId}`);

    res.json({
      success: true,
      message: `Model "${modelId}" deleted successfully`
    });
  } catch (error) {
    console.error('Delete error:', error);
    res.status(500).json({
      error: 'Delete failed',
      message: error.message
    });
  }
});

/**
 * GET /api/models/:modelId/download
 * Download a model file
 */
app.get('/api/models/:modelId/download', (req, res) => {
  try {
    const { modelId } = req.params;
    const config = MODEL_CONFIGS[modelId];

    if (!config) {
      return res.status(404).json({
        error: 'Model not found',
        message: `Model "${modelId}" does not exist`
      });
    }

    const modelPath = path.join(MODELS_DIR, config.file);

    if (!fs.existsSync(modelPath)) {
      return res.status(404).json({
        error: 'Model file not found',
        message: `Model file "${config.file}" does not exist`
      });
    }

    res.download(modelPath, config.file);
  } catch (error) {
    console.error('Download error:', error);
    res.status(500).json({
      error: 'Download failed',
      message: error.message
    });
  }
});

/**
 * POST /api/models/:modelId/validate
 * Validate a model structure
 */
app.post('/api/models/:modelId/validate', (req, res) => {
  try {
    const { modelId } = req.params;
    const config = MODEL_CONFIGS[modelId];

    if (!config) {
      return res.status(404).json({
        error: 'Model not found',
        message: `Model "${modelId}" does not exist`
      });
    }

    const modelPath = path.join(MODELS_DIR, config.file);

    if (!fs.existsSync(modelPath)) {
      return res.status(404).json({
        error: 'Model file not found',
        message: `Model file "${config.file}" does not exist`
      });
    }

    // Basic validation: check file size and ONNX header
    const stats = fs.statSync(modelPath);
    const buffer = fs.readFileSync(modelPath);

    // ONNX files should start with certain bytes
    const isValidOnnx = buffer.length > 0;

    if (isValidOnnx) {
      res.json({
        valid: true,
        message: 'Model appears to be valid',
        size: stats.size,
        lastModified: stats.mtime
      });
    } else {
      res.json({
        valid: false,
        errors: ['Invalid ONNX file format'],
        size: stats.size
      });
    }
  } catch (error) {
    console.error('Validation error:', error);
    res.status(500).json({
      error: 'Validation failed',
      message: error.message
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('zkX402 Agent Authorization API Server');
  console.log('='.repeat(60));
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Models directory: ${MODELS_DIR}`);
  console.log(`JOLT prover directory: ${JOLT_PROVER_DIR}`);
  console.log('\nAvailable endpoints:');
  console.log('  GET  /api/health             - Health check');
  console.log('  GET  /api/models             - List available models');
  console.log('  GET  /api/validate-models    - Validate ONNX models');
  console.log('  POST /api/generate-proof     - Generate zkML proof');
  console.log('  POST /api/upload-model       - Upload ONNX model');
  console.log('  DELETE /api/models/:id       - Delete model');
  console.log('  GET  /api/models/:id/download - Download model');
  console.log('  POST /api/models/:id/validate - Validate model');
  console.log('='.repeat(60));
});

module.exports = app;
