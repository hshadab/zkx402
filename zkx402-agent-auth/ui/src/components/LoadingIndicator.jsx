/**
 * Loading Indicator Component
 * Shows animated loading state during proof generation
 */

import React, { useState, useEffect } from 'react';

export default function LoadingIndicator({ message = 'Generating proof...', stage = null }) {
  const [dots, setDots] = useState('');
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    // Animate dots
    const dotsInterval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
    }, 500);

    // Track elapsed time
    const startTime = Date.now();
    const timeInterval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 100);

    return () => {
      clearInterval(dotsInterval);
      clearInterval(timeInterval);
    };
  }, []);

  const stages = [
    { name: 'Loading model', duration: 0.5 },
    { name: 'Preprocessing prover', duration: 1.0 },
    { name: 'Preparing inputs', duration: 0.2 },
    { name: 'Generating proof', duration: 2.0 },
    { name: 'Verifying proof', duration: 0.3 },
  ];

  const getCurrentStage = () => {
    let cumulative = 0;
    for (const s of stages) {
      cumulative += s.duration;
      if (elapsed < cumulative) {
        return s.name;
      }
    }
    return stages[stages.length - 1].name;
  };

  const getProgress = () => {
    const totalDuration = stages.reduce((sum, s) => sum + s.duration, 0);
    return Math.min((elapsed / totalDuration) * 100, 95); // Cap at 95% until done
  };

  return (
    <div className="bg-gray-900 rounded-lg p-8 border-2 border-green-500/50">
      {/* Spinner */}
      <div className="flex justify-center mb-6">
        <div className="relative">
          <div className="w-20 h-20 border-4 border-gray-700 rounded-full"></div>
          <div className="absolute top-0 left-0 w-20 h-20 border-4 border-green-500 rounded-full border-t-transparent animate-spin"></div>
        </div>
      </div>

      {/* Message */}
      <div className="text-center mb-6">
        <div className="text-2xl font-bold text-green-400 font-mono mb-2">
          {message}{dots}
        </div>
        <div className="text-sm text-gray-400 font-mono">
          {stage || getCurrentStage()}
        </div>
        <div className="text-xs text-gray-500 font-mono mt-2">
          {elapsed}s elapsed
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
        <div
          className="bg-gradient-to-r from-green-600 to-green-400 h-full transition-all duration-300"
          style={{ width: `${getProgress()}%` }}
        ></div>
      </div>

      {/* Stage indicators */}
      <div className="mt-6 flex justify-between">
        {stages.map((s, i) => {
          const isActive = getCurrentStage() === s.name;
          const isDone = stages.findIndex(stage => stage.name === getCurrentStage()) > i;

          return (
            <div key={i} className="flex-1 text-center">
              <div
                className={`w-3 h-3 mx-auto rounded-full mb-2 transition-colors ${
                  isDone
                    ? 'bg-green-500'
                    : isActive
                    ? 'bg-green-400 animate-pulse'
                    : 'bg-gray-700'
                }`}
              ></div>
              <div
                className={`text-xs font-mono transition-colors ${
                  isDone || isActive ? 'text-green-400' : 'text-gray-600'
                }`}
              >
                {s.name.split(' ')[0]}
              </div>
            </div>
          );
        })}
      </div>

      {/* Tips */}
      <div className="mt-6 p-4 bg-gray-800 rounded border border-gray-700">
        <div className="text-xs text-gray-400 font-mono space-y-1">
          <div>💡 <span className="text-gray-300">Proof generation typically takes 0.7-3s depending on model complexity</span></div>
          <div>🔒 <span className="text-gray-300">Your private data (balance, velocity) is never revealed in the proof</span></div>
          <div>✓ <span className="text-gray-300">Proof will be cryptographically verified before being returned</span></div>
        </div>
      </div>
    </div>
  );
}
