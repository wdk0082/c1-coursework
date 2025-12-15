// API Types for 5D Regression Backend

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  device: string;
  datasets_loaded: number;
  models_trained: number;
}

export interface UploadDatasetResponse {
  dataset_id: string;
  filename: string;
  num_samples: number;
  X_shape: number[];
  y_shape: number[];
  uploaded_at: string;
}

export interface TrainRequest {
  dataset_id: string;
  architecture?: {
    hidden_dims?: number[];
    dropout?: number;
    activation?: string;
  };
  training_params?: {
    learning_rate?: number;
    batch_size?: number;
    epochs?: number;
    weight_decay?: number;
  };
  split_ratios?: number[];
  standardize?: boolean;
  missing_strategy?: 'ignore' | 'mean' | 'median' | 'zero' | 'forward_fill';
}

export interface TrainResponse {
  model_id: string;
  best_epoch: number;
  best_val_loss: number;
  final_train_loss: number;
  test_metrics: {
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
  };
  training_time_seconds: number;
  training_memory_mb: number;
}

export interface PredictRequest {
  model_id: string;
  inputs: number[][];
}

export interface PredictResponse {
  model_id: string;
  predictions: number[];
  num_predictions: number;
  inference_memory_mb: number;
}

export interface Dataset {
  dataset_id: string;
  filename: string;
  num_samples: number;
  X_shape: number[];
  y_shape: number[];
  uploaded_at: string;
}

export interface DatasetsResponse {
  datasets: Dataset[];
  total: number;
}

export interface Model {
  model_id: string;
  dataset_id: string;
  best_epoch: number;
  best_val_loss: number;
  test_metrics: {
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
  };
  training_time_seconds: number;
  training_memory_mb: number;
  trained_at: string;
}

export interface ModelsResponse {
  models: Model[];
  total: number;
}

export interface ModelDetails {
  model_id: string;
  dataset_id: string;
  architecture: {
    hidden_dims: number[];
    dropout: number;
    activation: string;
  };
  training_params: {
    learning_rate: number;
    batch_size: number;
    epochs: number;
    weight_decay: number;
  };
  split_ratios: number[];
  standardize: boolean;
  missing_strategy: string;
  best_epoch: number;
  best_val_loss: number;
  test_metrics: {
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
  };
  training_time_seconds: number;
  training_memory_mb: number;
  trained_at: string;
}
