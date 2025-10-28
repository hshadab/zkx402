/**
 * Proof History Component
 * Displays history of generated proofs with export functionality
 */

import React, { useState, useEffect } from 'react';

export default function ProofHistory() {
  const [proofs, setProofs] = useState([]);
  const [filter, setFilter] = useState('all'); // 'all', 'approved', 'rejected'

  useEffect(() => {
    // Load proof history from localStorage
    const savedProofs = localStorage.getItem('proofHistory');
    if (savedProofs) {
      setProofs(JSON.parse(savedProofs));
    }

    // Listen for new proofs
    const handleNewProof = (event) => {
      const newProof = event.detail;
      const updatedProofs = [newProof, ...proofs].slice(0, 50); // Keep last 50
      setProofs(updatedProofs);
      localStorage.setItem('proofHistory', JSON.stringify(updatedProofs));
    };

    window.addEventListener('proof-generated', handleNewProof);
    return () => window.removeEventListener('proof-generated', handleNewProof);
  }, [proofs]);

  const filteredProofs = filter === 'all'
    ? proofs
    : proofs.filter(p => filter === 'approved' ? p.approved : !p.approved);

  const clearHistory = () => {
    if (confirm('Clear all proof history?')) {
      setProofs([]);
      localStorage.removeItem('proofHistory');
    }
  };

  const exportHistory = () => {
    const dataStr = JSON.stringify(proofs, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `zkx402-proof-history-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportProof = (proof) => {
    const dataStr = JSON.stringify(proof, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `zkx402-proof-${proof.id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-green-400 font-mono">
          Proof History
        </h2>
        <div className="flex gap-2">
          <button
            onClick={exportHistory}
            disabled={proofs.length === 0}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded font-mono text-sm transition-colors"
          >
            Export All
          </button>
          <button
            onClick={clearHistory}
            disabled={proofs.length === 0}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded font-mono text-sm transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Filter buttons */}
      <div className="flex gap-2 mb-4">
        {['all', 'approved', 'rejected'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded font-mono text-sm transition-colors ${
              filter === f
                ? 'bg-green-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
            {f !== 'all' && ` (${proofs.filter(p => f === 'approved' ? p.approved : !p.approved).length})`}
          </button>
        ))}
      </div>

      {/* Proof list */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {filteredProofs.length === 0 ? (
          <div className="text-center py-8 text-gray-500 font-mono">
            No proofs yet. Generate your first proof above!
          </div>
        ) : (
          filteredProofs.map((proof) => (
            <div
              key={proof.id}
              className="bg-gray-800 rounded p-4 border border-gray-700 hover:border-gray-600 transition-colors"
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-1 rounded text-xs font-mono ${
                      proof.approved
                        ? 'bg-green-900 text-green-300'
                        : 'bg-red-900 text-red-300'
                    }`}
                  >
                    {proof.approved ? '✓ APPROVED' : '✗ REJECTED'}
                  </span>
                  <span className="text-gray-400 font-mono text-sm">
                    {proof.model}
                  </span>
                </div>
                <button
                  onClick={() => exportProof(proof)}
                  className="text-blue-400 hover:text-blue-300 font-mono text-xs transition-colors"
                >
                  Export
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm font-mono">
                <div className="text-gray-400">
                  Proof Size: <span className="text-white">{proof.proofSize}</span>
                </div>
                <div className="text-gray-400">
                  Verification: <span className="text-white">{proof.verificationTime}</span>
                </div>
                <div className="text-gray-400">
                  Operations: <span className="text-white">{proof.operations}</span>
                </div>
                <div className="text-gray-400">
                  Output: <span className="text-white">{proof.output}</span>
                </div>
              </div>

              <div className="mt-2 text-xs text-gray-500 font-mono">
                {new Date(proof.timestamp).toLocaleString()}
              </div>

              {/* Inputs summary */}
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300 font-mono">
                  View Inputs
                </summary>
                <div className="mt-2 p-2 bg-gray-900 rounded text-xs font-mono space-y-1">
                  {Object.entries(proof.inputs).map(([key, value]) => (
                    <div key={key} className="text-gray-400">
                      {key}: <span className="text-white">{value}</span>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          ))
        )}
      </div>

      {proofs.length > 0 && (
        <div className="mt-4 text-center text-sm text-gray-500 font-mono">
          Showing {filteredProofs.length} of {proofs.length} proofs
        </div>
      )}
    </div>
  );
}
