'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api/client';
import type { ModelDetails } from '@/lib/api/types';

export default function ModelDetail() {
  const params = useParams();
  const router = useRouter();
  const modelId = params.id as string;

  const [model, setModel] = useState<ModelDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (modelId) {
      fetchModelDetails();
    }
  }, [modelId]);

  const fetchModelDetails = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getModelDetails(modelId);
      setModel(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load model details');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-600"></div>
          <p className="mt-2 text-gray-600">Loading model details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
        <button onClick={() => router.push('/models')} className="btn-secondary">
          Back to Models
        </button>
      </div>
    );
  }

  if (!model) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Model Details</h1>
        <button onClick={() => router.push('/models')} className="btn-secondary">
          Back to Models
        </button>
      </div>

      {/* Model ID */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-3">Model Information</h2>
        <div className="space-y-2">
          <div>
            <span className="text-gray-600">Model ID:</span>{' '}
            <span className="font-mono font-semibold">{model.model_id}</span>
          </div>
          <div>
            <span className="text-gray-600">Dataset ID:</span>{' '}
            <span className="font-mono">{model.dataset_id}</span>
          </div>
          <div>
            <span className="text-gray-600">Trained:</span>{' '}
            <span>{new Date(model.trained_at.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6')).toLocaleString()}</span>
          </div>
        </div>
      </div>

      {/* Architecture */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-3">Architecture</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Hidden Dimensions</p>
            <p className="font-semibold">[{model.architecture.hidden_dims.join(', ')}]</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Activation</p>
            <p className="font-semibold capitalize">{model.architecture.activation}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Dropout</p>
            <p className="font-semibold">{model.architecture.dropout}</p>
          </div>
        </div>
      </div>

      {/* Training Parameters */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-3">Training Parameters</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Learning Rate</p>
            <p className="font-semibold">{model.training_params.learning_rate}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Batch Size</p>
            <p className="font-semibold">{model.training_params.batch_size}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Epochs</p>
            <p className="font-semibold">{model.training_params.epochs}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Weight Decay</p>
            <p className="font-semibold">{model.training_params.weight_decay}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Best Epoch</p>
            <p className="font-semibold">{model.best_epoch}</p>
          </div>
        </div>
      </div>

      {/* Data Processing */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-3">Data Processing</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Split Ratios</p>
            <p className="font-semibold">[{model.split_ratios.join(', ')}]</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Standardize</p>
            <p className="font-semibold">{model.standardize ? 'Yes' : 'No'}</p>
          </div>
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-sm text-gray-600">Missing Strategy</p>
            <p className="font-semibold capitalize">{model.missing_strategy}</p>
          </div>
        </div>
      </div>

      {/* Test Metrics */}
      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-3">Performance Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Best Val Loss</p>
            <p className="text-2xl font-bold text-blue-600">{model.best_val_loss.toFixed(6)}</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Test MSE</p>
            <p className="text-2xl font-bold text-green-600">{model.test_metrics.mse.toFixed(6)}</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Test RMSE</p>
            <p className="text-2xl font-bold text-purple-600">{model.test_metrics.rmse.toFixed(6)}</p>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Test MAE</p>
            <p className="text-2xl font-bold text-orange-600">{model.test_metrics.mae.toFixed(6)}</p>
          </div>
          <div className="bg-teal-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Test R²</p>
            <p className="text-2xl font-bold text-teal-600">{model.test_metrics.r2.toFixed(6)}</p>
          </div>
          <div className="bg-indigo-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Training Time</p>
            <p className="text-2xl font-bold text-indigo-600">{model.training_time_seconds.toFixed(2)}s</p>
          </div>
          <div className="bg-pink-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Training Memory</p>
            <p className="text-2xl font-bold text-pink-600">{model.training_memory_mb.toFixed(2)} MB</p>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Actions</h2>
        <div className="flex space-x-3">
          <a href={`/predict?model=${model.model_id}`} className="btn-primary">
            Use for Predictions
          </a>
          <button onClick={() => router.push('/models')} className="btn-secondary">
            Back to Models List
          </button>
        </div>
      </div>
    </div>
  );
}
