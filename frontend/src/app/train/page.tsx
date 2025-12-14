'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api/client';
import type { DatasetsResponse, TrainRequest, TrainResponse } from '@/lib/api/types';

export default function TrainModel() {
  const [datasets, setDatasets] = useState<DatasetsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Form state
  const [datasetId, setDatasetId] = useState('');
  const [hiddenDims, setHiddenDims] = useState('64,32');
  const [dropout, setDropout] = useState('0.0');
  const [activation, setActivation] = useState('relu');
  const [learningRate, setLearningRate] = useState('0.001');
  const [batchSize, setBatchSize] = useState('32');
  const [epochs, setEpochs] = useState('100');
  const [weightDecay, setWeightDecay] = useState('0.0');
  const [splitRatios, setSplitRatios] = useState('0.7,0.15,0.15');
  const [standardize, setStandardize] = useState(true);
  const [missingStrategy, setMissingStrategy] = useState<'ignore' | 'mean' | 'median' | 'zero' | 'forward_fill'>('ignore');

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const data = await api.listDatasets();
      setDatasets(data);
      if (data.datasets.length > 0 && !datasetId) {
        setDatasetId(data.datasets[0].dataset_id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleTrain = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!datasetId) {
      setError('Please select a dataset');
      return;
    }

    try {
      setTraining(true);
      setError(null);
      setResult(null);

      const request: TrainRequest = {
        dataset_id: datasetId,
        architecture: {
          hidden_dims: hiddenDims.split(',').map(d => parseInt(d.trim())),
          dropout: parseFloat(dropout),
          activation,
        },
        training_params: {
          learning_rate: parseFloat(learningRate),
          batch_size: parseInt(batchSize),
          epochs: parseInt(epochs),
          weight_decay: parseFloat(weightDecay),
        },
        split_ratios: splitRatios.split(',').map(r => parseFloat(r.trim())),
        standardize,
        missing_strategy: missingStrategy,
      };

      const response = await api.trainModel(request);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Training failed');
      console.error(err);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Train Model</h1>

      {loading ? (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-600"></div>
        </div>
      ) : datasets && datasets.total === 0 ? (
        <div className="card">
          <p className="text-gray-600 mb-4">No datasets available. Please upload a dataset first.</p>
          <a href="/upload" className="btn-primary">
            Upload Dataset
          </a>
        </div>
      ) : (
        <form onSubmit={handleTrain}>
          {/* Dataset Selection */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Dataset</h2>
            <div>
              <label className="label">Select Dataset</label>
              <select
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                className="input-field"
                required
              >
                <option value="">-- Select a dataset --</option>
                {datasets?.datasets.map((ds) => (
                  <option key={ds.dataset_id} value={ds.dataset_id}>
                    {ds.filename} ({ds.num_samples} samples)
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Architecture Configuration */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Model Architecture</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Hidden Dimensions</label>
                <input
                  type="text"
                  value={hiddenDims}
                  onChange={(e) => setHiddenDims(e.target.value)}
                  className="input-field"
                  placeholder="64,32"
                  required
                />
                <p className="text-xs text-gray-500 mt-1">Comma-separated layer sizes</p>
              </div>

              <div>
                <label className="label">Activation Function</label>
                <select
                  value={activation}
                  onChange={(e) => setActivation(e.target.value)}
                  className="input-field"
                >
                  <option value="relu">ReLU</option>
                  <option value="tanh">Tanh</option>
                  <option value="sigmoid">Sigmoid</option>
                </select>
              </div>

              <div>
                <label className="label">Dropout</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={dropout}
                  onChange={(e) => setDropout(e.target.value)}
                  className="input-field"
                />
              </div>
            </div>
          </div>

          {/* Training Parameters */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Training Parameters</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  min="0"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  className="input-field"
                />
              </div>

              <div>
                <label className="label">Batch Size</label>
                <input
                  type="number"
                  min="1"
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value)}
                  className="input-field"
                />
              </div>

              <div>
                <label className="label">Epochs</label>
                <input
                  type="number"
                  min="1"
                  value={epochs}
                  onChange={(e) => setEpochs(e.target.value)}
                  className="input-field"
                />
              </div>

              <div>
                <label className="label">Weight Decay</label>
                <input
                  type="number"
                  step="0.00001"
                  min="0"
                  value={weightDecay}
                  onChange={(e) => setWeightDecay(e.target.value)}
                  className="input-field"
                />
              </div>
            </div>
          </div>

          {/* Data Processing */}
          <div className="card mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Data Processing</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Split Ratios (Train, Val, Test)</label>
                <input
                  type="text"
                  value={splitRatios}
                  onChange={(e) => setSplitRatios(e.target.value)}
                  className="input-field"
                  placeholder="0.7,0.15,0.15"
                />
              </div>

              <div>
                <label className="label">Missing Value Strategy</label>
                <select
                  value={missingStrategy}
                  onChange={(e) => setMissingStrategy(e.target.value as any)}
                  className="input-field"
                >
                  <option value="ignore">Ignore (remove rows)</option>
                  <option value="mean">Fill with mean</option>
                  <option value="median">Fill with median</option>
                  <option value="zero">Fill with zeros</option>
                  <option value="forward_fill">Forward fill</option>
                </select>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="standardize"
                  checked={standardize}
                  onChange={(e) => setStandardize(e.target.checked)}
                  className="h-4 w-4 text-blue-600 rounded"
                />
                <label htmlFor="standardize" className="ml-2 text-sm text-gray-700">
                  Standardize features (zero mean, unit variance)
                </label>
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
            <div className="card mb-6 bg-green-50 border-2 border-green-200">
              <h3 className="text-xl font-semibold text-green-800 mb-4">Training Complete!</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Model ID</p>
                  <p className="font-mono font-medium">{result.model_id}</p>
                </div>
                <div>
                  <p className="text-gray-600">Best Epoch</p>
                  <p className="font-medium">{result.best_epoch}</p>
                </div>
                <div>
                  <p className="text-gray-600">Best Val Loss</p>
                  <p className="font-medium">{result.best_val_loss.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Final Train Loss</p>
                  <p className="font-medium">{result.final_train_loss.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Test MSE</p>
                  <p className="font-medium">{result.test_metrics.mse.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Test RMSE</p>
                  <p className="font-medium">{result.test_metrics.rmse.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Test MAE</p>
                  <p className="font-medium">{result.test_metrics.mae.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Test R²</p>
                  <p className="font-medium">{result.test_metrics.r2.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Training Time</p>
                  <p className="font-medium">{result.training_time_seconds.toFixed(2)}s</p>
                </div>
                <div>
                  <p className="text-gray-600">Training Memory</p>
                  <p className="font-medium">{result.training_memory_mb.toFixed(2)} MB</p>
                </div>
              </div>
              <div className="mt-4">
                <a href="/predict" className="text-green-700 hover:text-green-800 font-medium underline">
                  Use this model for predictions →
                </a>
              </div>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={training}
            className="btn-primary w-full"
          >
            {training ? 'Training... This may take a while' : 'Start Training'}
          </button>
        </form>
      )}
    </div>
  );
}
