/**
 * Model Comparison Component
 * Compare authorization results across different models
 */

import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:3001/api';

export default function ModelComparison() {
  const [inputs, setInputs] = useState({
    amount: '50',
    balance: '1000',
    velocity_1h: '20',
    velocity_24h: '100',
    vendor_trust: '80'
  });

  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const models = ['simple_auth', 'neural_auth'];

  const runComparison = async () => {
    setLoading(true);
    setError(null);
    setResults({});

    try {
      // Run proofs in parallel
      const promises = models.map(model =>
        axios.post(`${API_BASE}/generate-proof`, {
          model,
          inputs
        }).then(res => ({ model, data: res.data }))
          .catch(err => ({ model, error: err.message }))
      );

      const responses = await Promise.all(promises);

      const resultsMap = {};
      responses.forEach(({ model, data, error }) => {
        resultsMap[model] = error ? { error } : data;
      });

      setResults(resultsMap);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <h2 className="text-2xl font-bold text-green-400 font-mono mb-6">
        Model Comparison
      </h2>

      {/* Shared inputs */}
      <div className="mb-6 p-4 bg-gray-800 rounded">
        <h3 className="text-lg font-bold text-white font-mono mb-4">
          Test Inputs
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(inputs).map(([key, value]) => (
            <div key={key}>
              <label className="block text-sm text-gray-400 font-mono mb-1">
                {key.replace('_', ' ')}
              </label>
              <input
                type="number"
                value={value}
                onChange={(e) => setInputs({ ...inputs, [key]: e.target.value })}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded text-white font-mono focus:outline-none focus:border-green-500"
              />
            </div>
          ))}
        </div>

        <button
          onClick={runComparison}
          disabled={loading}
          className="mt-4 w-full px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded font-mono font-bold transition-colors"
        >
          {loading ? 'Comparing Models...' : 'Run Comparison'}
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-700 rounded">
          <div className="text-red-400 font-mono text-sm">{error}</div>
        </div>
      )}

      {/* Results comparison */}
      {Object.keys(results).length > 0 && (
        <div className="grid md:grid-cols-2 gap-4">
          {models.map(model => {
            const result = results[model];

            if (!result) return null;

            if (result.error) {
              return (
                <div key={model} className="bg-gray-800 rounded p-4 border border-red-700">
                  <h4 className="text-lg font-bold text-white font-mono mb-2">
                    {model.replace('_', ' ').toUpperCase()}
                  </h4>
                  <div className="text-red-400 font-mono text-sm">{result.error}</div>
                </div>
              );
            }

            return (
              <div
                key={model}
                className={`bg-gray-800 rounded p-4 border-2 ${
                  result.approved ? 'border-green-600' : 'border-red-600'
                }`}
              >
                <div className="flex justify-between items-start mb-4">
                  <h4 className="text-lg font-bold text-white font-mono">
                    {model.replace('_', ' ').toUpperCase()}
                  </h4>
                  <span
                    className={`px-3 py-1 rounded font-mono text-sm ${
                      result.approved
                        ? 'bg-green-900 text-green-300'
                        : 'bg-red-900 text-red-300'
                    }`}
                  >
                    {result.approved ? '✓ APPROVED' : '✗ REJECTED'}
                  </span>
                </div>

                <div className="space-y-2 text-sm font-mono">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Output:</span>
                    <span className="text-white">{result.output}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Proof Size:</span>
                    <span className="text-white">{result.proofSize}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Verification:</span>
                    <span className={result.verification ? 'text-green-400' : 'text-red-400'}>
                      {result.verification ? '✓ Valid' : '✗ Invalid'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Verify Time:</span>
                    <span className="text-white">{result.verificationTime}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Operations:</span>
                    <span className="text-white">{result.operations}</span>
                  </div>
                </div>

                {/* Proof snippet */}
                <details className="mt-4">
                  <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300 font-mono">
                    View Proof Data
                  </summary>
                  <div className="mt-2 p-2 bg-gray-900 rounded text-xs font-mono overflow-auto max-h-32">
                    <div className="text-gray-400">Commitment:</div>
                    <div className="text-green-400 break-all">
                      {result.zkmlProof.commitment.substring(0, 64)}...
                    </div>
                  </div>
                </details>
              </div>
            );
          })}
        </div>
      )}

      {/* Insights */}
      {Object.keys(results).length > 0 && !error && (
        <div className="mt-6 p-4 bg-blue-900/30 border border-blue-700 rounded">
          <h4 className="text-sm font-bold text-blue-400 font-mono mb-2">
            📊 Comparison Insights
          </h4>
          <div className="text-xs text-gray-300 font-mono space-y-1">
            {(() => {
              const approved = models.filter(m => results[m]?.approved);
              const rejected = models.filter(m => results[m] && !results[m]?.approved);

              if (approved.length === models.length) {
                return <div>✓ All models approved this transaction - high confidence authorization</div>;
              } else if (rejected.length === models.length) {
                return <div>✗ All models rejected this transaction - clear policy violation</div>;
              } else {
                return (
                  <>
                    <div>⚠️ Mixed results - models disagree on authorization</div>
                    <div>Approved: {approved.join(', ') || 'none'}</div>
                    <div>Rejected: {rejected.join(', ') || 'none'}</div>
                  </>
                );
              }
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
