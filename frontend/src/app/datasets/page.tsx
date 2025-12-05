'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api/client';
import type { DatasetsResponse } from '@/lib/api/types';

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.listDatasets();
      setDatasets(data);
    } catch (err) {
      setError('Failed to load datasets');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Datasets</h1>
        <div className="flex space-x-3">
          <button
            onClick={fetchDatasets}
            className="btn-secondary"
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <a href="/upload" className="btn-primary">
            Upload New Dataset
          </a>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      {loading && !datasets ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-gray-200 border-t-blue-600"></div>
          <p className="mt-2 text-gray-600">Loading datasets...</p>
        </div>
      ) : datasets && datasets.total === 0 ? (
        <div className="card text-center py-12">
          <div className="text-6xl mb-4">📦</div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No Datasets</h2>
          <p className="text-gray-600 mb-6">Upload your first dataset to get started</p>
          <a href="/upload" className="btn-primary">
            Upload Dataset
          </a>
        </div>
      ) : datasets ? (
        <>
          <div className="card mb-6">
            <div className="text-sm text-gray-600">
              Total datasets: <span className="font-semibold">{datasets.total}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4">
            {datasets.datasets.map((dataset) => (
              <div key={dataset.dataset_id} className="card hover:shadow-lg transition-shadow">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-800 mb-2">
                      {dataset.filename}
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-600">Dataset ID</p>
                        <p className="font-mono text-xs break-all">{dataset.dataset_id}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Samples</p>
                        <p className="font-semibold">{dataset.num_samples.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">X Shape</p>
                        <p className="font-mono text-xs">[{dataset.X_shape.join(', ')}]</p>
                      </div>
                      <div>
                        <p className="text-gray-600">y Shape</p>
                        <p className="font-mono text-xs">[{dataset.y_shape.join(', ')}]</p>
                      </div>
                    </div>
                    <div className="mt-3 text-xs text-gray-500">
                      Uploaded: {new Date(dataset.uploaded_at.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3T$4:$5:$6')).toLocaleString()}
                    </div>
                  </div>
                  <div className="ml-4">
                    <a
                      href={`/train?dataset=${dataset.dataset_id}`}
                      className="btn-primary text-sm"
                    >
                      Train Model
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
