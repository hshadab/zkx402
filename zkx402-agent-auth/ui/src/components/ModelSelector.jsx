import React from 'react'
import { getFeaturedModels } from '../utils/curatedModels'

export default function ModelSelector({ selectedModel, onModelChange }) {
  // Use featured models only (4 advanced models)
  const FEATURED_MODELS = getFeaturedModels()

  const getSelectedModel = () => FEATURED_MODELS.find(m => m.id === selectedModel)
  const selected = getSelectedModel()

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="text-xl font-bold">Select Agent Model to Verify</h2>
        <p className="text-sm text-gray-400 mt-1">
          Choose from our 4 advanced zkML authorization policies
        </p>
      </div>

      {/* Selected Model Info */}
      {selected && (
        <div className="mb-6 p-4 sm:p-6 rounded-lg bg-gradient-to-r from-accent-blue/10 to-accent-purple/10 border border-accent-blue/30">
          <div className="flex flex-col sm:flex-row items-start justify-between gap-3">
            <div className="flex-1">
              <h3 className="font-semibold text-lg text-accent-blue mb-1">
                {selected.name}
              </h3>
              <p className="text-sm text-gray-300 mb-2">{selected.description}</p>
              <p className="text-xs text-gray-400">
                <span className="font-medium">Use Case:</span> {selected.useCase}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-3 sm:gap-4 mt-3 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
              {selected.operations}
            </span>
            <span className="hidden sm:inline">•</span>
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              {selected.proofTime}
            </span>
            <span className="hidden sm:inline">•</span>
            <span className="flex items-center gap-1">
              <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              {selected.inputs.length} inputs
            </span>
          </div>

          {/* Example Scenarios */}
          {selected.examples && selected.examples.length > 0 && (
            <div className="mt-4 pt-4 border-t border-accent-blue/20">
              <h4 className="text-xs font-semibold text-gray-300 mb-3 flex items-center gap-2">
                <svg className="w-4 h-4 text-accent-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                How it works - Example scenarios:
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {selected.examples.map((example, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border text-xs ${
                      example.expected === 'approved' || example.expected === 'true'
                        ? 'bg-accent-green/5 border-accent-green/30'
                        : 'bg-red-500/5 border-red-500/30'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      {example.expected === 'approved' || example.expected === 'true' ? (
                        <svg className="w-4 h-4 text-accent-green flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                      <span className={`font-semibold ${
                        example.expected === 'approved' || example.expected === 'true'
                          ? 'text-accent-green'
                          : 'text-red-400'
                      }`}>
                        {example.desc}
                      </span>
                    </div>
                    <div className="pl-6 space-y-1 text-gray-400">
                      {Object.entries(example).map(([key, value]) => {
                        if (key === 'expected' || key === 'desc') return null
                        return (
                          <div key={key} className="flex justify-between">
                            <span className="opacity-70">{key}:</span>
                            <span className="font-mono">{value}</span>
                          </div>
                        )
                      })}
                      <div className="flex justify-between pt-1 border-t border-gray-700/50 mt-1">
                        <span className="opacity-70">Result:</span>
                        <span className={`font-semibold ${
                          example.expected === 'approved' || example.expected === 'true'
                            ? 'text-accent-green'
                            : 'text-red-400'
                        }`}>
                          {example.expected}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Grid - 4 cards in a row */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {FEATURED_MODELS.map(model => {
          const isSelected = selectedModel === model.id

          return (
            <button
              key={model.id}
              onClick={() => onModelChange(model.id)}
              className={`
                p-4 rounded-lg border-2 text-left transition-all transform hover:scale-105
                ${isSelected
                  ? 'border-accent-blue bg-accent-blue/10 shadow-lg'
                  : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                }
              `}
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className={`font-semibold ${
                  isSelected ? 'text-accent-blue' : 'text-white'
                }`}>
                  {model.name}
                </h3>
                <svg className="w-5 h-5 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                </svg>
              </div>

              <p className="text-sm text-gray-400 mb-3 line-clamp-2">{model.description}</p>

              <div className="flex flex-wrap gap-1 mb-3">
                {model.inputs.slice(0, 3).map(input => (
                  <span key={input} className="px-2 py-1 bg-dark-600 rounded text-xs text-gray-400">
                    {input}
                  </span>
                ))}
                {model.inputs.length > 3 && (
                  <span className="px-2 py-1 bg-dark-600 rounded text-xs text-gray-400">
                    +{model.inputs.length - 3}
                  </span>
                )}
              </div>

              <div className="flex gap-3 text-xs text-gray-500">
                <span>{model.operations}</span>
                <span>•</span>
                <span>{model.proofTime}</span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
