import { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, DollarSign, Activity, Clock, CheckCircle, XCircle } from 'lucide-react';

export default function Analytics() {
  const [stats, setStats] = useState(null);
  const [modelBreakdown, setModelBreakdown] = useState(null);
  const [blockchainStats, setBlockchainStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchAnalytics = async () => {
    try {
      const [statsRes, modelsRes, blockchainRes] = await Promise.all([
        fetch('/api/analytics/stats'),
        fetch('/api/analytics/models'),
        fetch('/api/analytics/blockchain')
      ]);

      if (statsRes.ok && modelsRes.ok) {
        setStats(await statsRes.json());
        setModelBreakdown(await modelsRes.json());
      }

      if (blockchainRes.ok) {
        setBlockchainStats(await blockchainRes.json());
      }
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatUptime = (seconds) => {
    if (!seconds) return '0s';
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent-green"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">Analytics Dashboard</h1>
        <p className="text-gray-400">Real-time metrics and usage statistics for zkX402</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={<Activity className="w-5 h-5" />}
          label="Total Requests"
          value={stats?.totalRequests || 0}
          subtext={`${stats?.requests24h || 0} in last 24h`}
          color="blue"
        />
        <StatCard
          icon={<DollarSign className="w-5 h-5" />}
          label="Blockchain Revenue"
          value={`$${blockchainStats?.totalRevenue || '0.00'}`}
          subtext={`$${blockchainStats?.revenue24h || '0.00'} in last 24h`}
          color="green"
        />
        <StatCard
          icon={<CheckCircle className="w-5 h-5" />}
          label="Wallet Balance"
          value={`$${blockchainStats?.currentBalance || '0.00'}`}
          subtext={`${blockchainStats?.totalTransactions || 0} total transactions`}
          color="emerald"
        />
        <StatCard
          icon={<Clock className="w-5 h-5" />}
          label="Avg Response Time"
          value={`${stats?.avgResponseTime || 0}ms`}
          subtext={`Uptime: ${formatUptime(stats?.uptime)}`}
          color="purple"
        />
      </div>

      {/* Model Usage Breakdown */}
      <div className="bg-dark-800 rounded-lg shadow p-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-300" />
          <h2 className="text-xl font-semibold text-white">Model Usage</h2>
        </div>

        {modelBreakdown && Object.keys(modelBreakdown).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-dark-700">
              <thead>
                <tr className="bg-dark-700">
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Model ID
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Total Requests
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Success Rate
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Paid Requests
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                    Avg Response Time
                  </th>
                </tr>
              </thead>
              <tbody className="bg-dark-800 divide-y divide-dark-700">
                {Object.entries(modelBreakdown)
                  .sort((a, b) => b[1].totalRequests - a[1].totalRequests)
                  .map(([modelId, data]) => (
                    <tr key={modelId} className="hover:bg-dark-700">
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-white">
                        {modelId}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                        {data.totalRequests}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          parseFloat(data.successRate) > 90 ? 'bg-green-900 text-green-300' :
                          parseFloat(data.successRate) > 70 ? 'bg-yellow-900 text-yellow-300' :
                          'bg-red-900 text-red-300'
                        }`}>
                          {data.successRate}%
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                        {data.paidRequests}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                        {data.avgResponseTime}ms
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            No model usage data available yet
          </div>
        )}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Requests */}
        <div className="bg-dark-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Requests</h3>
          {stats?.recentRequests && stats.recentRequests.length > 0 ? (
            <div className="space-y-2">
              {stats.recentRequests.map((req) => (
                <div key={req.id} className="flex items-center justify-between p-3 bg-dark-700 rounded">
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white">
                      {req.endpoint}
                    </div>
                    <div className="text-xs text-gray-400">
                      {req.modelId || 'No model'} • {new Date(req.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  {req.success ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-400" />
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-gray-400">No recent requests</div>
          )}
        </div>

        {/* Recent Blockchain Payments */}
        <div className="bg-dark-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Blockchain Payments</h3>
          {blockchainStats?.recentTransactions && blockchainStats.recentTransactions.length > 0 ? (
            <div className="space-y-2">
              {blockchainStats.recentTransactions.map((tx) => (
                <div key={tx.id} className="flex items-center justify-between p-3 bg-dark-700 rounded">
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white">
                      ${tx.amountUSDC} USDC
                    </div>
                    <div className="text-xs text-gray-400">
                      From: {tx.from.substring(0, 6)}...{tx.from.substring(38)} • {new Date(tx.timestamp).toLocaleTimeString()}
                    </div>
                    <a
                      href={`https://basescan.org/tx/${tx.txHash}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-accent-green hover:underline"
                    >
                      View on BaseScan
                    </a>
                  </div>
                  <CheckCircle className="w-5 h-5 text-green-400" />
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-4 text-gray-400">No recent payments</div>
          )}
        </div>
      </div>

      {/* Payment Wallet Info */}
      <div className="bg-gradient-to-r from-dark-800 to-dark-700 rounded-lg shadow p-6 border border-accent-green/20">
        <h3 className="text-lg font-semibold text-white mb-3">Payment Wallet</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Address:</span>
            <code className="text-sm font-mono bg-dark-900 text-accent-green px-2 py-1 rounded">
              0x1f409E94684804e5158561090Ced8941B47B0CC6
            </code>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Network:</span>
            <span className="text-sm font-medium text-white">Base Mainnet (Chain ID: 8453)</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Token:</span>
            <span className="text-sm font-medium text-white">USDC (6 decimals)</span>
          </div>
          <div className="mt-4">
            <a
              href="https://basescan.org/address/0x1f409E94684804e5158561090Ced8941B47B0CC6"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-accent-green hover:text-accent-green/80"
            >
              <TrendingUp className="w-4 h-4" />
              View on BaseScan
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, subtext, color }) {
  const colors = {
    blue: 'bg-blue-900/50 text-blue-400',
    green: 'bg-green-900/50 text-green-400',
    emerald: 'bg-emerald-900/50 text-emerald-400',
    purple: 'bg-purple-900/50 text-purple-400',
  };

  return (
    <div className="bg-dark-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-3">
        <div className={`p-2 rounded-lg ${colors[color]}`}>
          {icon}
        </div>
      </div>
      <div className="space-y-1">
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-sm font-medium text-gray-400">{label}</div>
        {subtext && <div className="text-xs text-gray-500">{subtext}</div>}
      </div>
    </div>
  );
}
