'use client';

import { useState } from 'react';
import { api } from '@/lib/api/client';
import type { UploadDatasetResponse } from '@/lib/api/types';

export default function UploadDataset() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadDatasetResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const validExtensions = ['.npz', '.pkl', '.pickle'];
      const hasValidExt = validExtensions.some(ext => selectedFile.name.endsWith(ext));
      if (!hasValidExt) {
        setError('Please select a .npz or .pkl file');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    try {
      setUploading(true);
      setError(null);
      const response = await api.uploadDataset(file);
      setResult(response);
      setFile(null);
      // Reset file input
      const fileInput = document.getElementById('file-input') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload dataset');
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-800 mb-6">Upload Dataset</h1>

      <div className="card mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Dataset Requirements
        </h2>
        <div className="space-y-2 text-sm text-gray-600">
          <p>• File format: <span className="font-mono bg-gray-100 px-2 py-1 rounded">.npz</span> or <span className="font-mono bg-gray-100 px-2 py-1 rounded">.pkl</span></p>
          <p>• Required keys: <span className="font-mono bg-gray-100 px-2 py-1 rounded">X</span> and <span className="font-mono bg-gray-100 px-2 py-1 rounded">y</span></p>
          <p>• X shape: <span className="font-mono bg-gray-100 px-2 py-1 rounded">(n_samples, 5)</span> - 5D input features</p>
          <p>• y shape: <span className="font-mono bg-gray-100 px-2 py-1 rounded">(n_samples,)</span> - 1D output targets</p>
          <p className="mt-3 text-xs">
            Alternative key names are also supported: inputs/outputs or features/targets
          </p>
        </div>
      </div>

      <div className="card">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Upload File
        </h2>

        <div className="mb-4">
          <label htmlFor="file-input" className="label">
            Select .npz or .pkl file
          </label>
          <input
            id="file-input"
            type="file"
            accept=".npz,.pkl,.pickle"
            onChange={handleFileChange}
            className="input-field"
            disabled={uploading}
          />
          {file && (
            <p className="mt-2 text-sm text-gray-600">
              Selected: <span className="font-medium">{file.name}</span> ({(file.size / 1024).toFixed(2)} KB)
            </p>
          )}
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
            {error}
          </div>
        )}

        {result && (
          <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg mb-4">
            <div className="font-semibold mb-2">Dataset uploaded successfully!</div>
            <div className="text-sm space-y-1">
              <p>Dataset ID: <span className="font-mono">{result.dataset_id}</span></p>
              <p>Filename: {result.filename}</p>
              <p>Samples: {result.num_samples}</p>
              <p>X shape: [{result.X_shape.join(', ')}]</p>
              <p>y shape: [{result.y_shape.join(', ')}]</p>
            </div>
            <div className="mt-3">
              <a href="/train" className="text-green-700 hover:text-green-800 font-medium underline">
                Train a model with this dataset →
              </a>
            </div>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="btn-primary w-full"
        >
          {uploading ? 'Uploading...' : 'Upload Dataset'}
        </button>
      </div>

      {/* Example Code */}
      <div className="card mt-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">
          Example: Creating a Dataset
        </h2>
        <pre className="bg-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
{`import numpy as np
import pickle

# Create sample data
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]  # Linear combination

# Save as .npz
np.savez('my_dataset.npz', X=X, y=y)

# Or save as .pkl
with open('my_dataset.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)`}
        </pre>
      </div>
    </div>
  );
}
