/**
 * Analytics Manager for zkX402
 * Tracks API usage, payments, and generates metrics
 */

class AnalyticsManager {
  constructor() {
    // In-memory storage (can be replaced with database)
    this.requests = [];
    this.payments = [];
    this.startTime = new Date();
  }

  /**
   * Log an API request
   */
  logRequest(data) {
    const entry = {
      id: `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      endpoint: data.endpoint,
      method: data.method,
      modelId: data.modelId,
      success: data.success,
      responseTime: data.responseTime,
      errorMessage: data.errorMessage,
      userAgent: data.userAgent,
      ip: data.ip,
      hasPaidHeader: data.hasPaidHeader,
    };

    this.requests.push(entry);

    // Keep only last 1000 requests to prevent memory overflow
    if (this.requests.length > 1000) {
      this.requests.shift();
    }

    return entry;
  }

  /**
   * Log a payment verification
   */
  logPayment(data) {
    const entry = {
      id: `pay_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      txHash: data.txHash,
      sender: data.sender,
      amount: data.amount,
      amountUSDC: data.amountUSDC,
      modelId: data.modelId,
      verified: data.verified,
      errorMessage: data.errorMessage,
    };

    this.payments.push(entry);

    // Keep only last 1000 payments
    if (this.payments.length > 1000) {
      this.payments.shift();
    }

    return entry;
  }

  /**
   * Get summary statistics
   */
  getStats() {
    const now = new Date();
    const last24h = new Date(now - 24 * 60 * 60 * 1000);
    const last7d = new Date(now - 7 * 24 * 60 * 60 * 1000);

    // Filter requests by time period
    const requests24h = this.requests.filter(r => new Date(r.timestamp) > last24h);
    const requests7d = this.requests.filter(r => new Date(r.timestamp) > last7d);

    // Filter payments by time period
    const payments24h = this.payments.filter(p => new Date(p.timestamp) > last24h);
    const payments7d = this.payments.filter(p => new Date(p.timestamp) > last7d);

    // Calculate revenue
    const totalRevenue = this.payments
      .filter(p => p.verified)
      .reduce((sum, p) => sum + parseFloat(p.amountUSDC || 0), 0);

    const revenue24h = payments24h
      .filter(p => p.verified)
      .reduce((sum, p) => sum + parseFloat(p.amountUSDC || 0), 0);

    const revenue7d = payments7d
      .filter(p => p.verified)
      .reduce((sum, p) => sum + parseFloat(p.amountUSDC || 0), 0);

    // Count requests by endpoint
    const endpointCounts = {};
    this.requests.forEach(r => {
      endpointCounts[r.endpoint] = (endpointCounts[r.endpoint] || 0) + 1;
    });

    // Count requests by model
    const modelCounts = {};
    this.requests.forEach(r => {
      if (r.modelId) {
        modelCounts[r.modelId] = (modelCounts[r.modelId] || 0) + 1;
      }
    });

    // Success rate
    const successfulRequests = this.requests.filter(r => r.success).length;
    const successRate = this.requests.length > 0
      ? (successfulRequests / this.requests.length * 100).toFixed(1)
      : 0;

    // Payment success rate
    const verifiedPayments = this.payments.filter(p => p.verified).length;
    const paymentSuccessRate = this.payments.length > 0
      ? (verifiedPayments / this.payments.length * 100).toFixed(1)
      : 0;

    // Average response time
    const responseTimes = this.requests
      .filter(r => r.responseTime)
      .map(r => r.responseTime);
    const avgResponseTime = responseTimes.length > 0
      ? (responseTimes.reduce((sum, t) => sum + t, 0) / responseTimes.length).toFixed(0)
      : 0;

    return {
      uptime: Math.floor((now - this.startTime) / 1000), // seconds
      totalRequests: this.requests.length,
      requests24h: requests24h.length,
      requests7d: requests7d.length,
      successRate: parseFloat(successRate),
      avgResponseTime: parseInt(avgResponseTime),

      totalPayments: this.payments.length,
      payments24h: payments24h.length,
      payments7d: payments7d.length,
      verifiedPayments,
      paymentSuccessRate: parseFloat(paymentSuccessRate),

      totalRevenue: totalRevenue.toFixed(4),
      revenue24h: revenue24h.toFixed(4),
      revenue7d: revenue7d.toFixed(4),

      endpointCounts,
      modelCounts,

      recentRequests: this.requests.slice(-10).reverse(),
      recentPayments: this.payments.slice(-10).reverse(),
    };
  }

  /**
   * Get model usage breakdown
   */
  getModelBreakdown() {
    const breakdown = {};

    this.requests.forEach(r => {
      if (!r.modelId) return;

      if (!breakdown[r.modelId]) {
        breakdown[r.modelId] = {
          totalRequests: 0,
          successfulRequests: 0,
          paidRequests: 0,
          totalResponseTime: 0,
        };
      }

      breakdown[r.modelId].totalRequests++;
      if (r.success) breakdown[r.modelId].successfulRequests++;
      if (r.hasPaidHeader) breakdown[r.modelId].paidRequests++;
      if (r.responseTime) breakdown[r.modelId].totalResponseTime += r.responseTime;
    });

    // Calculate averages
    Object.keys(breakdown).forEach(modelId => {
      const data = breakdown[modelId];
      data.successRate = data.totalRequests > 0
        ? ((data.successfulRequests / data.totalRequests) * 100).toFixed(1)
        : 0;
      data.avgResponseTime = data.successfulRequests > 0
        ? (data.totalResponseTime / data.successfulRequests).toFixed(0)
        : 0;
    });

    return breakdown;
  }

  /**
   * Get time-series data for charts
   */
  getTimeSeries(hours = 24) {
    const now = new Date();
    const intervals = [];
    const requestCounts = [];
    const paymentCounts = [];

    // Create hourly buckets
    for (let i = hours - 1; i >= 0; i--) {
      const intervalStart = new Date(now - i * 60 * 60 * 1000);
      const intervalEnd = new Date(now - (i - 1) * 60 * 60 * 1000);

      intervals.push(intervalStart.toISOString());

      const requestsInInterval = this.requests.filter(r => {
        const reqTime = new Date(r.timestamp);
        return reqTime >= intervalStart && reqTime < intervalEnd;
      }).length;

      const paymentsInInterval = this.payments.filter(p => {
        const payTime = new Date(p.timestamp);
        return payTime >= intervalStart && payTime < intervalEnd && p.verified;
      }).length;

      requestCounts.push(requestsInInterval);
      paymentCounts.push(paymentsInInterval);
    }

    return {
      intervals,
      requestCounts,
      paymentCounts,
    };
  }

  /**
   * Clear all analytics data (for testing)
   */
  clear() {
    this.requests = [];
    this.payments = [];
    this.startTime = new Date();
  }
}

module.exports = new AnalyticsManager();
