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
  const [inputType, setInputType] = useState<'single' | 'multiple'>('single');
  const [input1, setInput1] = useState('');
  const [input2, setInput2] = useState('');
  const [input3, setInput3] = useState('');
  const [input4, setInput4] = useState('');
  const [input5, setInput5] = useState('');
  const [multipleInputs, setMultipleInputs] = useState('');

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

      let inputs: number[][];

      if (inputType === 'single') {
        inputs = [[
          parseFloat(input1),
          parseFloat(input2),
          parseFloat(input3),
          parseFloat(input4),
          parseFloat(input5),
        ]];
      } else {
        // Parse multiple inputs (one vector per line)
        const lines = multipleInputs.trim().split('\n');
        inputs = lines.map(line => {
          const values = line.trim().split(/[,\s]+/).map(v => parseFloat(v));
          if (values.length !== 5) {
            throw new Error(`Each input must have exactly 5 values, got ${values.length}`);
          }
          return values;
        });
      }

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

          {/* Input Type Selection */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Input Type</h2>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="single"
                  checked={inputType === 'single'}
                  onChange={(e) => setInputType('single')}
                  className="h-4 w-4 text-blue-600"
                />
                <span className="ml-2">Single Input</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="multiple"
                  checked={inputType === 'multiple'}
                  onChange={(e) => setInputType('multiple')}
                  className="h-4 w-4 text-blue-600"
                />
                <span className="ml-2">Multiple Inputs</span>
              </label>
            </div>
          </div>

          {/* Input Fields */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Input Values</h2>

            {inputType === 'single' ? (
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
            ) : (
              <div>
                <label className="label">
                  Multiple Inputs (one vector per line, 5 values each)
                </label>
                <textarea
                  value={multipleInputs}
                  onChange={(e) => setMultipleInputs(e.target.value)}
                  className="input-field font-mono"
                  rows={8}
                  placeholder="1.0 2.0 3.0 4.0 5.0&#10;0.5 1.5 2.5 3.5 4.5&#10;-1.0 0.0 1.0 2.0 3.0"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">
                  Separate values with spaces or commas. One input vector per line.
                </p>
              </div>
            )}
          </div>

          {/* Error/Success Messages */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
              {error}
            </div>
          )}

          {result && (
            <div className="card mb-6 bg-blue-50 border-2 border-blue-200">
              <h3 className="text-xl font-semibold text-blue-800 mb-4">Predictions</h3>
              <div className="space-y-3">
                <div>
                  <p className="text-sm text-gray-600">Model ID</p>
                  <p className="font-mono text-sm">{result.model_id}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-2">
                    Results ({result.num_predictions} prediction{result.num_predictions > 1 ? 's' : ''})
                  </p>
                  <div className="bg-white rounded-lg p-4 max-h-64 overflow-y-auto">
                    {result.predictions.map((pred, idx) => (
                      <div key={idx} className="flex items-center justify-between py-2 border-b last:border-b-0">
                        <span className="text-gray-600">Prediction {idx + 1}:</span>
                        <span className="font-semibold text-lg">{pred.toFixed(6)}</span>
                      </div>
                    ))}
                  </div>
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
