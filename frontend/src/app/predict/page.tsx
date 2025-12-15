'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api/client';
import type { ModelsResponse, PredictRequest, PredictResponse } from '@/lib/api/types';

export default function Predict() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [modelId, setModelId] = useState('');
  const [input1, setInput1] = useState('');
  const [input2, setInput2] = useState('');
  const [input3, setInput3] = useState('');
  const [input4, setInput4] = useState('');
  const [input5, setInput5] = useState('');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const data = await api.listModels();
      setModels(data);
      if (data.models.length > 0 && !modelId) {
        setModelId(data.models[0].model_id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!modelId) {
      setError('Please select a model');
      return;
    }

    try {
      setPredicting(true);
      setError(null);
      setResult(null);

      const inputs: number[][] = [[
        parseFloat(input1),
        parseFloat(input2),
        parseFloat(input3),
        parseFloat(input4),
        parseFloat(input5),
      ]];

      const request: PredictRequest = {
        model_id: modelId,
        inputs,
      };

      const response = await api.predict(request);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed');
      console.error(err);
    } finally {
      setPredicting(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Make Predictions</h1>

      {loading ? (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-600"></div>
        </div>
      ) : models && models.total === 0 ? (
        <div className="card">
          <p className="text-gray-600 mb-4">No trained models available. Please train a model first.</p>
          <a href="/train" className="btn-primary">
            Train Model
          </a>
        </div>
      ) : (
        <form onSubmit={handlePredict}>
          {/* Model Selection */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Select Model</h2>
            <div>
              <label className="label">Trained Model</label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="input-field"
                required
              >
                <option value="">-- Select a model --</option>
                {models?.models.map((model) => (
                  <option key={model.model_id} value={model.model_id}>
                    {model.model_id} (RMSE: {model.test_metrics.rmse.toFixed(4)})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Input Fields */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Input Values (5 Features)</h2>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div>
                <label className="label">Feature 1</label>
                <input
                  type="number"
                  step="any"
                  value={input1}
                  onChange={(e) => setInput1(e.target.value)}
                  className="input-field"
                  placeholder="0.0"
                  required
                />
              </div>
              <div>
                <label className="label">Feature 2</label>
                <input
                  type="number"
                  step="any"
                  value={input2}
                  onChange={(e) => setInput2(e.target.value)}
                  className="input-field"
                  placeholder="0.0"
                  required
                />
              </div>
              <div>
                <label className="label">Feature 3</label>
                <input
                  type="number"
                  step="any"
                  value={input3}
                  onChange={(e) => setInput3(e.target.value)}
                  className="input-field"
                  placeholder="0.0"
                  required
                />
              </div>
              <div>
                <label className="label">Feature 4</label>
                <input
                  type="number"
                  step="any"
                  value={input4}
                  onChange={(e) => setInput4(e.target.value)}
                  className="input-field"
                  placeholder="0.0"
                  required
                />
              </div>
              <div>
                <label className="label">Feature 5</label>
                <input
                  type="number"
                  step="any"
                  value={input5}
                  onChange={(e) => setInput5(e.target.value)}
                  className="input-field"
                  placeholder="0.0"
                  required
                />
              </div>
            </div>
          </div>

          {/* Error/Success Messages */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
              {error}
            </div>
          )}

          {result && (
            <div className="card mb-6 bg-blue-50 border-2 border-blue-200">
              <h3 className="text-xl font-semibold text-blue-800 mb-4">Prediction Result</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-600">Model ID</p>
                  <p className="font-mono text-sm">{result.model_id}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Predicted Value</p>
                  <p className="font-semibold text-2xl text-blue-800">{result.predictions[0].toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Inference Memory</p>
                  <p className="font-medium text-sm">{result.inference_memory_mb.toFixed(4)} MB</p>
                </div>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={predicting}
            className="btn-primary w-full"
          >
            {predicting ? 'Predicting...' : 'Make Prediction'}
          </button>
        </form>
      )}
    </div>
  );
}
