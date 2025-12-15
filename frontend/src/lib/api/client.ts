import axios from 'axios';
import type {
  HealthResponse,
  UploadDatasetResponse,
  TrainRequest,
  TrainResponse,
  PredictRequest,
  PredictResponse,
  DatasetsResponse,
  ModelsResponse,
  ModelDetails,
} from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health check
  health: async (): Promise<HealthResponse> => {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },

  // Upload dataset
  uploadDataset: async (file: File): Promise<UploadDatasetResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<UploadDatasetResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Train model
  trainModel: async (request: TrainRequest): Promise<TrainResponse> => {
    const response = await apiClient.post<TrainResponse>('/train', request);
    return response.data;
  },

  // Make predictions
  predict: async (request: PredictRequest): Promise<PredictResponse> => {
    const response = await apiClient.post<PredictResponse>('/predict', request);
    return response.data;
  },

  // List datasets
  listDatasets: async (): Promise<DatasetsResponse> => {
    const response = await apiClient.get<DatasetsResponse>('/datasets');
    return response.data;
  },

  // List models
  listModels: async (): Promise<ModelsResponse> => {
    const response = await apiClient.get<ModelsResponse>('/models');
    return response.data;
  },

  // Get model details
  getModelDetails: async (modelId: string): Promise<ModelDetails> => {
    const response = await apiClient.get<ModelDetails>(`/models/${modelId}`);
    return response.data;
  },
};

export default api;
