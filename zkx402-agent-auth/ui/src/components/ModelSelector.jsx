import React, { useState } from 'react'
import { CURATED_MODELS, MODEL_CATEGORIES, getAllCategories } from '../utils/curatedModels'

export default function ModelSelector({ selectedModel, onModelChange }) {
  const [selectedCategory, setSelectedCategory] = useState('All')

  const categories = ['All', ...getAllCategories()]

  const filteredModels = selectedCategory === 'All'
    ? CURATED_MODELS
    : CURATED_MODELS.filter(m => m.category === selectedCategory)

  const getSelectedModel = () => CURATED_MODELS.find(m => m.id === selectedModel)
  const selected = getSelectedModel()

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="text-xl font-bold">Select Authorization Model</h2>
        <p className="text-sm text-gray-400 mt-1">
          Choose from 10 production-ready models for x402 payment authorization
        </p>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              selectedCategory === cat
                ? 'bg-accent-blue text-white'
                : 'bg-dark-700 text-gray-400 hover:bg-dark-600'
            }`}
          >
            {cat === 'All' ? '📋 All Models' : `${MODEL_CATEGORIES[cat]?.icon || ''} ${cat}`}
            <span className="ml-2 text-xs opacity-70">
              ({cat === 'All' ? CURATED_MODELS.length : CURATED_MODELS.filter(m => m.category === cat).length})
            </span>
          </button>
        ))}
      </div>

      {/* Selected Model Info */}
      {selected && (
        <div className="mb-6 p-4 rounded-lg bg-gradient-to-r from-accent-blue/10 to-accent-purple/10 border border-accent-blue/30">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="font-semibold text-lg text-accent-blue mb-1">
                {selected.name}
              </h3>
              <p className="text-sm text-gray-300 mb-2">{selected.description}</p>
              <p className="text-xs text-gray-400">
                <span className="font-medium">Use Case:</span> {selected.useCase}
              </p>
            </div>
            <div className="text-right">
              <span className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${
                selected.category === 'Basic' ? 'bg-accent-green/20 text-accent-green' :
                selected.category === 'Velocity' ? 'bg-accent-blue/20 text-accent-blue' :
                selected.category === 'Access' ? 'bg-accent-purple/20 text-accent-purple' :
                'bg-accent-orange/20 text-accent-orange'
              }`}>
                {selected.category}
              </span>
            </div>
          </div>
          <div className="flex gap-4 mt-3 text-xs text-gray-500">
            <span>📝 {selected.operations}</span>
            <span>•</span>
            <span>⚡ {selected.proofTime}</span>
            <span>•</span>
            <span>🔢 {selected.inputs.length} inputs</span>
          </div>
        </div>
      )}

      {/* Model Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredModels.map(model => {
          const isSelected = selectedModel === model.id
          const categoryInfo = MODEL_CATEGORIES[model.category]

          return (
            <button
              key={model.id}
              onClick={() => onModelChange(model.id)}
              className={`
                p-4 rounded-lg border-2 text-left transition-all transform hover:scale-105
                ${isSelected
                  ? `border-${categoryInfo?.color || 'accent-blue'} bg-${categoryInfo?.color || 'accent-blue'}/10 shadow-lg`
                  : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                }
              `}
            >
              <div className="flex items-start justify-between mb-2">
                <h3 className={`font-semibold ${
                  isSelected ? `text-${categoryInfo?.color || 'accent-blue'}` : 'text-white'
                }`}>
                  {model.name}
                </h3>
                <span className="text-lg">{categoryInfo?.icon || '📊'}</span>
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

      {filteredModels.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No models found in this category</p>
        </div>
      )}

      {/* Legend */}
      <div className="mt-6 pt-6 border-t border-dark-600">
        <h4 className="text-sm font-medium text-gray-400 mb-3">Model Categories</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {Object.entries(MODEL_CATEGORIES).map(([name, info]) => (
            <div key={name} className="flex items-center gap-2">
              <span className="text-lg">{info.icon}</span>
              <div>
                <div className="text-sm font-medium text-white">{name}</div>
                <div className="text-xs text-gray-500">{info.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
