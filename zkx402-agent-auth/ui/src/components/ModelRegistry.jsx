/**
 * Model Registry Component
 * Upload and manage ONNX authorization models
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:3001/api';

export default function ModelRegistry() {
  const [models, setModels] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/models`);
      setModels(response.data.models);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.onnx')) {
      setUploadError('Only .onnx files are allowed');
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    const formData = new FormData();
    formData.append('model', file);
    formData.append('description', `Uploaded model: ${file.name}`);

    try {
      const response = await axios.post(`${API_BASE}/upload-model`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setUploadSuccess(`Successfully uploaded: ${file.name}`);
      fetchModels(); // Refresh model list
      event.target.value = ''; // Clear file input
    } catch (error) {
      setUploadError(error.response?.data?.message || error.message);
    } finally {
      setUploading(false);
    }
  };

  const deleteModel = async (modelId) => {
    if (!confirm(`Delete model "${modelId}"? This cannot be undone.`)) {
      return;
    }

    try {
      await axios.delete(`${API_BASE}/models/${modelId}`);
      setUploadSuccess(`Model "${modelId}" deleted successfully`);
      fetchModels();
    } catch (error) {
      setUploadError(error.response?.data?.message || error.message);
    }
  };

  const downloadModel = async (modelId, filename) => {
    try {
      const response = await axios.get(`${API_BASE}/models/${modelId}/download`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      setUploadError(`Failed to download: ${error.message}`);
    }
  };

  const validateModel = async (modelId) => {
    try {
      const response = await axios.post(`${API_BASE}/models/${modelId}/validate`);
      if (response.data.valid) {
        setUploadSuccess(`Model "${modelId}" is valid!`);
      } else {
        setUploadError(`Model "${modelId}" validation failed: ${response.data.errors.join(', ')}`);
      }
    } catch (error) {
      setUploadError(`Validation failed: ${error.message}`);
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <h2 className="text-2xl font-bold text-green-400 font-mono mb-6">
        Model Registry
      </h2>

      {/* Upload section */}
      <div className="mb-6 p-4 bg-gray-800 rounded border-2 border-dashed border-gray-700 hover:border-green-600 transition-colors">
        <div className="text-center">
          <div className="mb-4">
            <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <label className="cursor-pointer">
            <span className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded font-mono font-bold transition-colors inline-block">
              {uploading ? 'Uploading...' : 'Upload ONNX Model'}
            </span>
            <input
              type="file"
              accept=".onnx"
              onChange={handleFileUpload}
              disabled={uploading}
              className="hidden"
            />
          </label>
          <p className="mt-2 text-sm text-gray-400 font-mono">
            Only .onnx files accepted
          </p>
        </div>
      </div>

      {/* Status messages */}
      {uploadError && (
        <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded">
          <div className="text-red-400 font-mono text-sm">{uploadError}</div>
          <button
            onClick={() => setUploadError(null)}
            className="mt-2 text-red-300 hover:text-red-200 text-xs underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {uploadSuccess && (
        <div className="mb-4 p-3 bg-green-900/30 border border-green-700 rounded">
          <div className="text-green-400 font-mono text-sm">{uploadSuccess}</div>
          <button
            onClick={() => setUploadSuccess(null)}
            className="mt-2 text-green-300 hover:text-green-200 text-xs underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Model list */}
      <div className="space-y-3">
        <h3 className="text-lg font-bold text-white font-mono mb-4">
          Available Models ({models.length})
        </h3>

        {models.length === 0 ? (
          <div className="text-center py-8 text-gray-500 font-mono">
            No models found. Upload your first model above!
          </div>
        ) : (
          models.map((model) => (
            <div
              key={model.id}
              className="bg-gray-800 rounded p-4 border border-gray-700 hover:border-gray-600 transition-colors"
            >
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h4 className="text-lg font-bold text-white font-mono">
                    {model.id}
                  </h4>
                  <p className="text-sm text-gray-400 font-mono mt-1">
                    {model.description}
                  </p>
                </div>
                <span
                  className={`px-2 py-1 rounded text-xs font-mono ${
                    model.available
                      ? 'bg-green-900 text-green-300'
                      : 'bg-red-900 text-red-300'
                  }`}
                >
                  {model.available ? '● Available' : '○ Missing'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-sm font-mono mb-3">
                <div className="text-gray-400">
                  File: <span className="text-white">{model.file}</span>
                </div>
                <div className="text-gray-400">
                  Inputs: <span className="text-white">{model.inputCount}</span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => validateModel(model.id)}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded font-mono text-xs transition-colors"
                >
                  Validate
                </button>
                <button
                  onClick={() => downloadModel(model.id, model.file)}
                  disabled={!model.available}
                  className="px-3 py-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white rounded font-mono text-xs transition-colors"
                >
                  Download
                </button>
                <button
                  onClick={() => deleteModel(model.id)}
                  className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded font-mono text-xs transition-colors ml-auto"
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Info box */}
      <div className="mt-6 p-4 bg-blue-900/30 border border-blue-700 rounded">
        <h4 className="text-sm font-bold text-blue-400 font-mono mb-2">
          📝 Model Requirements
        </h4>
        <div className="text-xs text-gray-300 font-mono space-y-1">
          <div>• Must be ONNX format (.onnx file)</div>
          <div>• Should accept 5 inputs: [amount, balance, velocity_1h, velocity_24h, vendor_trust]</div>
          <div>• All operations must be supported by JOLT Atlas</div>
          <div>• Supported ops: Add, Sub, Mul, Greater, Less, GreaterEqual, MatMul, etc.</div>
          <div>• Use integer scaling (no floating point)</div>
        </div>
      </div>
    </div>
  );
}
