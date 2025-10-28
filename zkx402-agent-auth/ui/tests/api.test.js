/**
 * End-to-End API Tests for zkX402 Agent Authorization
 *
 * Tests the full flow: UI → API → JOLT Atlas → Verification
 */

const axios = require('axios');
const { exec } = require('child_process');
const { promisify } = require('util');
const path = require('path');

const execAsync = promisify(exec);

const API_BASE = process.env.API_URL || 'http://localhost:3001';
const TIMEOUT = 120000; // 2 minutes for proof generation

describe('zkX402 Agent Authorization E2E Tests', () => {
  beforeAll(async () => {
    // Check if server is running
    try {
      await axios.get(`${API_BASE}/api/health`);
    } catch (error) {
      throw new Error('Server not running! Start with: npm run server');
    }
  });

  describe('Health Check', () => {
    test('should return healthy status', async () => {
      const response = await axios.get(`${API_BASE}/api/health`);
      expect(response.status).toBe(200);
      expect(response.data.status).toBe('ok');
      expect(response.data.modelsAvailable).toBeGreaterThan(0);
    });
  });

  describe('Model Discovery', () => {
    test('should list available models', async () => {
      const response = await axios.get(`${API_BASE}/api/models`);
      expect(response.status).toBe(200);
      expect(response.data.models).toBeInstanceOf(Array);
      expect(response.data.models.length).toBeGreaterThan(0);

      // Check model structure
      const model = response.data.models[0];
      expect(model).toHaveProperty('id');
      expect(model).toHaveProperty('file');
      expect(model).toHaveProperty('description');
      expect(model).toHaveProperty('available');
    });

    test('should validate models exist', async () => {
      const response = await axios.get(`${API_BASE}/api/validate-models`);
      expect(response.status).toBe(200);
      expect(response.data.valid).toBe(true);
    });
  });

  describe('Proof Generation - Simple Auth', () => {
    test('should approve valid transaction', async () => {
      const payload = {
        model: 'simple_auth',
        inputs: {
          amount: '50',      // $0.50 (scaled: 50)
          balance: '1000',   // $10.00 (scaled: 1000)
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '80'
        }
      };

      const response = await axios.post(`${API_BASE}/api/generate-proof`, payload);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('approved');
      expect(response.data.approved).toBe(true);
      expect(response.data).toHaveProperty('output');
      expect(response.data).toHaveProperty('verification');
      expect(response.data.verification).toBe(true);
      expect(response.data).toHaveProperty('proofSize');
      expect(response.data).toHaveProperty('zkmlProof');
      expect(response.data.zkmlProof).toHaveProperty('commitment');
      expect(response.data.zkmlProof).toHaveProperty('response');
      expect(response.data.zkmlProof).toHaveProperty('evaluation');
    }, TIMEOUT);

    test('should reject transaction with excessive amount', async () => {
      const payload = {
        model: 'simple_auth',
        inputs: {
          amount: '200',     // $2.00 (>10% of balance)
          balance: '1000',   // $10.00
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '80'
        }
      };

      const response = await axios.post(`${API_BASE}/api/generate-proof`, payload);

      expect(response.status).toBe(200);
      expect(response.data.approved).toBe(false);
      expect(response.data.verification).toBe(true);
    }, TIMEOUT);

    test('should reject transaction with low trust score', async () => {
      const payload = {
        model: 'simple_auth',
        inputs: {
          amount: '50',
          balance: '1000',
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '30'  // Below 50 threshold
        }
      };

      const response = await axios.post(`${API_BASE}/api/generate-proof`, payload);

      expect(response.status).toBe(200);
      expect(response.data.approved).toBe(false);
    }, TIMEOUT);
  });

  describe('Proof Generation - Neural Network', () => {
    test('should process neural network authorization', async () => {
      const payload = {
        model: 'neural_auth',
        inputs: {
          amount: '50',
          balance: '1000',
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '75'
        }
      };

      const response = await axios.post(`${API_BASE}/api/generate-proof`, payload);

      expect(response.status).toBe(200);
      expect(response.data).toHaveProperty('approved');
      expect(response.data).toHaveProperty('verification');
      expect(response.data.verification).toBe(true);
      expect(response.data).toHaveProperty('operations');
      expect(response.data.operations).toBeGreaterThan(20);
    }, TIMEOUT);
  });

  describe('Error Handling', () => {
    test('should return error for invalid model', async () => {
      const payload = {
        model: 'nonexistent_model',
        inputs: {
          amount: '50',
          balance: '1000',
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '80'
        }
      };

      try {
        await axios.post(`${API_BASE}/api/generate-proof`, payload);
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.response.status).toBe(400);
      }
    });

    test('should return error for missing inputs', async () => {
      const payload = {
        model: 'simple_auth',
        inputs: {}
      };

      try {
        await axios.post(`${API_BASE}/api/generate-proof`, payload);
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.response.status).toBeGreaterThanOrEqual(400);
      }
    });

    test('should return error for missing model parameter', async () => {
      const payload = {
        inputs: {
          amount: '50',
          balance: '1000',
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '80'
        }
      };

      try {
        await axios.post(`${API_BASE}/api/generate-proof`, payload);
        fail('Should have thrown an error');
      } catch (error) {
        expect(error.response.status).toBe(400);
      }
    });
  });

  describe('Performance', () => {
    test('proof generation should complete in reasonable time', async () => {
      const payload = {
        model: 'simple_auth',
        inputs: {
          amount: '50',
          balance: '1000',
          velocity_1h: '20',
          velocity_24h: '100',
          vendor_trust: '80'
        }
      };

      const start = Date.now();
      const response = await axios.post(`${API_BASE}/api/generate-proof`, payload);
      const duration = Date.now() - start;

      expect(response.status).toBe(200);
      expect(duration).toBeLessThan(10000); // Should complete within 10 seconds
    }, TIMEOUT);
  });
});
