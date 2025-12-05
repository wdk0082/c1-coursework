'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api/client';
import type { HealthResponse } from '@/lib/api/types';

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.health();
      setHealth(data);
    } catch (err) {
      setError('Failed to connect to backend. Make sure the API is running on http://localhost:8000');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          5D Regression Platform
        </h1>
        <p className="text-gray-600">
          Train and deploy neural network models for 5-dimensional to 1-dimensional regression
        </p>
      </div>

      {/* System Status Card */}
      <div className="card mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-gray-800">System Status</h2>
          <button
            onClick={fetchHealth}
            className="btn-secondary text-sm"
            disabled={loading}
          >
            {loading ? 'Checking...' : 'Refresh'}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        {loading && !health ? (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-600"></div>
            <p className="mt-2 text-gray-600">Connecting to backend...</p>
          </div>
        ) : health ? (
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Status</div>
              <div className="text-2xl font-semibold text-green-600 capitalize">
                {health.status}
              </div>
            </div>

            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Device</div>
              <div className="text-2xl font-semibold text-blue-600 uppercase">
                {health.device}
              </div>
            </div>

            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Datasets Loaded</div>
              <div className="text-2xl font-semibold text-purple-600">
                {health.datasets_loaded}
              </div>
            </div>

            <div className="bg-orange-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600 mb-1">Models Trained</div>
              <div className="text-2xl font-semibold text-orange-600">
                {health.models_trained}
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="/upload"
            className="block p-6 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors border-2 border-blue-200"
          >
            <div className="text-3xl mb-2">📤</div>
            <h3 className="font-semibold text-lg mb-1">Upload Dataset</h3>
            <p className="text-sm text-gray-600">
              Upload a .npz file containing your training data
            </p>
          </a>

          <a
            href="/train"
            className="block p-6 bg-green-50 hover:bg-green-100 rounded-lg transition-colors border-2 border-green-200"
          >
            <div className="text-3xl mb-2">🚀</div>
            <h3 className="font-semibold text-lg mb-1">Train Model</h3>
            <p className="text-sm text-gray-600">
              Configure and train a neural network model
            </p>
          </a>

          <a
            href="/predict"
            className="block p-6 bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors border-2 border-purple-200"
          >
            <div className="text-3xl mb-2">🎯</div>
            <h3 className="font-semibold text-lg mb-1">Make Predictions</h3>
            <p className="text-sm text-gray-600">
              Use trained models to predict new values
            </p>
          </a>
        </div>
      </div>
    </div>
  );
}
