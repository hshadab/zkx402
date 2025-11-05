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
    { name: 'Loading model', duration: 2 },
    { name: 'Preprocessing prover', duration: 3 },
    { name: 'Preparing inputs', duration: 1 },
    { name: 'Generating proof', duration: 8 },      // 5-10 seconds actual
    { name: 'Verifying proof', duration: 240 },     // 40s-7.5min actual (use 4min average)
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
    <div className="bg-gray-900 rounded-lg p-4 sm:p-8 border-2 border-green-500/50">
      {/* Spinner */}
      <div className="flex justify-center mb-4 sm:mb-6">
        <div className="relative">
          <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-gray-700 rounded-full"></div>
          <div className="absolute top-0 left-0 w-16 h-16 sm:w-20 sm:h-20 border-4 border-green-500 rounded-full border-t-transparent animate-spin"></div>
        </div>
      </div>

      {/* Message */}
      <div className="text-center mb-4 sm:mb-6">
        <div className="text-lg sm:text-2xl font-bold text-green-400 font-mono mb-2">
          {message}{dots}
        </div>
        <div className="text-xs sm:text-sm text-gray-400 font-mono">
          {stage || getCurrentStage()}
        </div>
        <div className="text-xs text-gray-500 font-mono mt-2">
          {elapsed}s elapsed
          {elapsed > 60 && (
            <span className="ml-2 text-green-400">
              ({Math.floor(elapsed / 60)}m {elapsed % 60}s)
            </span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
        <div
          className="bg-gradient-to-r from-green-600 to-green-400 h-full transition-all duration-300"
          style={{ width: `${getProgress()}%` }}
        ></div>
      </div>

      {/* Stage indicators - hidden on mobile to save space */}
      <div className="mt-4 sm:mt-6 hidden sm:flex justify-between">
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

      {/* Tips - mobile optimized */}
      <div className="mt-4 sm:mt-6 p-3 sm:p-4 bg-gray-800 rounded border border-gray-700">
        <div className="text-xs text-gray-400 font-mono space-y-1.5 sm:space-y-1">
          <div className="flex items-start gap-2">
            <span className="flex-shrink-0">⏱️</span>
            <span className="text-gray-300">Proof generation + verification takes 1-8 minutes</span>
          </div>
          <div className="flex items-start gap-2">
            <span className="flex-shrink-0">🔒</span>
            <span className="text-gray-300">Your private data is never revealed</span>
          </div>
          <div className="flex items-start gap-2 hidden sm:flex">
            <span className="flex-shrink-0">✓</span>
            <span className="text-gray-300">Comprehensive cryptographic verification ensures proof validity</span>
          </div>
        </div>
      </div>
    </div>
  );
}
