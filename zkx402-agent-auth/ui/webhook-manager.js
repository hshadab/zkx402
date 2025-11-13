/**
 * Webhook Manager for zkX402
 * Allows agents to register webhooks for async proof completion notifications
 */

const axios = require('axios');
const logger = require('./logger');
const CONSTANTS = require('./constants');

class WebhookManager {
  constructor() {
    // In-memory storage (in production, use a database)
    this.webhooks = new Map();
    this.proofRequests = new Map();
  }

  /**
   * Validate webhook URL
   * @throws {Error} if URL is invalid or not allowed
   */
  validateWebhookUrl(callbackUrl) {
    try {
      const url = new URL(callbackUrl);

      // Only allow HTTP/HTTPS
      if (!['http:', 'https:'].includes(url.protocol)) {
        throw new Error('Only HTTP and HTTPS protocols are allowed');
      }

      // Block internal/private IP addresses to prevent SSRF
      const hostname = url.hostname.toLowerCase();

      // Block localhost
      if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1') {
        throw new Error('Localhost URLs are not allowed');
      }

      // Block private IP ranges
      if (hostname.startsWith('192.168.') ||
          hostname.startsWith('10.') ||
          hostname.startsWith('172.16.') || hostname.startsWith('172.17.') ||
          hostname.startsWith('172.18.') || hostname.startsWith('172.19.') ||
          hostname.startsWith('172.20.') || hostname.startsWith('172.21.') ||
          hostname.startsWith('172.22.') || hostname.startsWith('172.23.') ||
          hostname.startsWith('172.24.') || hostname.startsWith('172.25.') ||
          hostname.startsWith('172.26.') || hostname.startsWith('172.27.') ||
          hostname.startsWith('172.28.') || hostname.startsWith('172.29.') ||
          hostname.startsWith('172.30.') || hostname.startsWith('172.31.')) {
        throw new Error('Private network URLs are not allowed');
      }

      // Block metadata endpoints
      if (hostname === '169.254.169.254') {
        throw new Error('Cloud metadata URLs are not allowed');
      }

      return true;
    } catch (error) {
      if (error instanceof TypeError) {
        throw new Error('Invalid URL format');
      }
      throw error;
    }
  }

  /**
   * Register a webhook for proof completion notifications
   */
  registerWebhook(webhookId, callbackUrl, metadata = {}) {
    // Validate URL before registering
    this.validateWebhookUrl(callbackUrl);

    logger.info('Registering webhook', { webhookId, callbackUrl });

    this.webhooks.set(webhookId, {
      id: webhookId,
      callbackUrl,
      metadata,
      createdAt: new Date().toISOString(),
      deliveries: [],
    });
    return this.webhooks.get(webhookId);
  }

  /**
   * Get webhook by ID
   */
  getWebhook(webhookId) {
    return this.webhooks.get(webhookId);
  }

  /**
   * Delete webhook
   */
  deleteWebhook(webhookId) {
    return this.webhooks.delete(webhookId);
  }

  /**
   * List all webhooks
   */
  listWebhooks() {
    return Array.from(this.webhooks.values());
  }

  /**
   * Trigger webhook with proof result
   */
  async triggerWebhook(webhookId, proofResult) {
    const webhook = this.webhooks.get(webhookId);
    if (!webhook) {
      logger.error('Webhook not found', { webhookId });
      return null;
    }

    const payload = {
      event: 'proof.completed',
      webhook_id: webhookId,
      timestamp: new Date().toISOString(),
      data: proofResult,
    };

    const delivery = {
      id: `delivery_${Date.now()}`,
      timestamp: new Date().toISOString(),
      status: 'pending',
      payload,
    };

    try {
      const response = await axios.post(webhook.callbackUrl, payload, {
        headers: {
          'Content-Type': 'application/json',
          'X-zkX402-Event': 'proof.completed',
          'X-zkX402-Webhook-ID': webhookId,
        },
        timeout: CONSTANTS.WEBHOOK.TIMEOUT_MS,
      });

      delivery.status = 'delivered';
      delivery.response_status = response.status;
      webhook.deliveries.push(delivery);

      logger.info('Webhook delivered successfully', { webhookId, responseStatus: response.status });
      return delivery;
    } catch (error) {
      delivery.status = 'failed';
      delivery.error = error.message;
      webhook.deliveries.push(delivery);

      logger.error('Webhook delivery failed', { webhookId, error: error.message });
      return delivery;
    }
  }

  /**
   * Create a proof request with webhook
   */
  createProofRequest(requestId, webhookId, policyId, inputs) {
    this.proofRequests.set(requestId, {
      id: requestId,
      webhookId,
      policyId,
      inputs,
      status: 'pending',
      createdAt: new Date().toISOString(),
    });
    return this.proofRequests.get(requestId);
  }

  /**
   * Complete a proof request and trigger webhook
   */
  async completeProofRequest(requestId, proofResult) {
    const request = this.proofRequests.get(requestId);
    if (!request) {
      logger.error('Proof request not found', { requestId });
      return null;
    }

    request.status = 'completed';
    request.completedAt = new Date().toISOString();
    request.result = proofResult;

    // Trigger webhook if configured
    if (request.webhookId) {
      await this.triggerWebhook(request.webhookId, {
        request_id: requestId,
        policy_id: request.policyId,
        inputs: request.inputs,
        ...proofResult,
      });
    }

    return request;
  }
}

module.exports = new WebhookManager();
