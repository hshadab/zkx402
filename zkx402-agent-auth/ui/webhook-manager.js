/**
 * Webhook Manager for zkX402
 * Allows agents to register webhooks for async proof completion notifications
 */

const axios = require('axios');

class WebhookManager {
  constructor() {
    // In-memory storage (in production, use a database)
    this.webhooks = new Map();
    this.proofRequests = new Map();
  }

  /**
   * Register a webhook for proof completion notifications
   */
  registerWebhook(webhookId, callbackUrl, metadata = {}) {
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
      console.error(`Webhook not found: ${webhookId}`);
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
        timeout: 10000,
      });

      delivery.status = 'delivered';
      delivery.response_status = response.status;
      webhook.deliveries.push(delivery);

      console.log(`Webhook delivered successfully: ${webhookId}`);
      return delivery;
    } catch (error) {
      delivery.status = 'failed';
      delivery.error = error.message;
      webhook.deliveries.push(delivery);

      console.error(`Webhook delivery failed: ${webhookId}`, error.message);
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
      console.error(`Proof request not found: ${requestId}`);
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
