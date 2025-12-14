'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api/client';
import type { ModelsResponse } from '@/lib/api/types';
import Link from 'next/link';

export default function Models() {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.listModels();
      setModels(data);
    } catch (err) {
      setError('Failed to load models');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Trained Models</h1>
        <div className="flex space-x-3">
          <button
            onClick={fetchModels}
            className="btn-secondary"
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <a href="/train" className="btn-primary">
            Train New Model
          </a>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {loading && !models ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-600"></div>
          <p className="mt-2 text-gray-600">Loading models...</p>
        </div>
      ) : models && models.total === 0 ? (
        <div className="card text-center py-12">
          <div className="text-6xl mb-4">🤖</div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No Trained Models</h2>
          <p className="text-gray-600 mb-6">Train your first model to get started</p>
          <a href="/train" className="btn-primary">
            Train Model
          </a>
        </div>
      ) : models ? (
        <>
          <div className="card mb-6">
            <div className="text-sm text-gray-600">
              Total models: <span className="font-semibold">{models.total}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4">
            {models.models.map((model) => (
              <div key={model.model_id} className="card hover:shadow-lg transition-shadow">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-800 mb-3">
                      {model.model_id}
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
                      <div>
                        <p className="text-gray-600">Dataset ID</p>
                        <p className="font-mono text-xs break-all">{model.dataset_id}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Best Epoch</p>
                        <p className="font-semibold">{model.best_epoch}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Best Val Loss</p>
                        <p className="font-semibold">{model.best_val_loss.toFixed(6)}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Trained</p>
                        <p className="text-xs">{new Date(model.trained_at.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6')).toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
                      <div>
                        <span className="text-gray-600">MSE:</span>{' '}
                        <span className="font-semibold">{model.test_metrics.mse.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">RMSE:</span>{' '}
                        <span className="font-semibold">{model.test_metrics.rmse.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">MAE:</span>{' '}
                        <span className="font-semibold">{model.test_metrics.mae.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">R²:</span>{' '}
                        <span className="font-semibold">{model.test_metrics.r2.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Time:</span>{' '}
                        <span className="font-semibold">{model.training_time_seconds.toFixed(2)}s</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Memory:</span>{' '}
                        <span className="font-semibold">{model.training_memory_mb.toFixed(2)} MB</span>
                      </div>
                    </div>
                  </div>
                  <div className="ml-4 flex flex-col space-y-2">
                    <Link
                      href={`/models/${model.model_id}`}
                      className="btn-secondary text-sm text-center"
                    >
                      View Details
                    </Link>
                    <a
                      href={`/predict?model=${model.model_id}`}
                      className="btn-primary text-sm text-center"
                    >
                      Use for Prediction
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
