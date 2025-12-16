Test Structure
==============

This document explains the organization and purpose of each test module in the suite.

Test Directory Layout
---------------------

.. code-block:: text

   backend/test/
   ├── conftest.py         # Shared fixtures
   ├── test_api.py         # FastAPI endpoint tests
   ├── test_data.py        # Data loading tests
   ├── test_model.py       # Neural network model tests
   └── test_trainer.py     # Trainer tests

API Tests (test_api.py)
-----------------------

Tests for FastAPI endpoints using the ``TestClient``.

**TestHealthEndpoint**
   - ``test_health_check``: Verifies health endpoint returns correct status
   - ``test_health_shows_counts``: Checks dataset/model counts in response

**TestUploadEndpoint**
   - ``test_upload_valid_npz``: Tests successful NPZ file upload
   - ``test_upload_wrong_extension``: Verifies rejection of invalid file types
   - ``test_upload_wrong_feature_dimensions``: Tests validation of feature shape
   - ``test_upload_creates_dataset_entry``: Confirms dataset is stored

**TestDatasetsEndpoint**
   - ``test_list_empty_datasets``: Tests empty dataset list
   - ``test_list_datasets_after_upload``: Verifies datasets appear after upload

**TestTrainEndpoint**
   - ``test_train_nonexistent_dataset``: Tests 404 for missing dataset
   - ``test_train_minimal_config``: Tests training with minimal configuration
   - ``test_train_custom_architecture``: Tests custom model architecture
   - ``test_train_creates_model_entry``: Confirms model is stored

**TestPredictEndpoint**
   - ``test_predict_nonexistent_model``: Tests 404 for missing model
   - ``test_predict_single_sample``: Tests single input prediction
   - ``test_predict_multiple_samples``: Tests batch prediction
   - ``test_predict_wrong_dimensions``: Tests input validation

**TestModelsEndpoint**
   - ``test_list_empty_models``: Tests empty model list
   - ``test_list_models_after_training``: Verifies models appear after training
   - ``test_get_model_details``: Tests model detail retrieval
   - ``test_get_nonexistent_model``: Tests 404 for missing model

**TestIntegration**
   - ``test_full_workflow``: End-to-end test of upload -> train -> predict
   - ``test_multiple_models``: Tests training multiple models on same dataset

Data Tests (test_data.py)
-------------------------

Tests for the ``load_data`` function and data preprocessing.

**TestLoadData**
   - ``test_load_npz_file``: Tests loading data from .npz file
   - ``test_load_npz_with_splits``: Tests loading with train/val/test splits
   - ``test_load_npz_alternative_keys``: Tests alternative key names (inputs/outputs)
   - ``test_load_pkl_file``: Tests loading data from .pkl file
   - ``test_invalid_file_extension``: Tests rejection of invalid file types
   - ``test_missing_file``: Tests error for non-existent files
   - ``test_wrong_feature_dimensions``: Tests validation of 5D features
   - ``test_sample_count_mismatch``: Tests X/y sample count validation
   - ``test_standardization_applied``: Tests feature standardization
   - ``test_split_ratios``: Tests train/val/test split ratios

**TestMissingValues**
   - ``test_ignore_strategy``: Tests removing rows with NaN
   - ``test_mean_strategy``: Tests filling NaN with column mean
   - ``test_median_strategy``: Tests filling NaN with column median
   - ``test_zero_strategy``: Tests filling NaN with zeros
   - ``test_invalid_strategy``: Tests error for unknown strategy

**TestStandardization**
   - ``test_standardize_features``: Tests zero mean and unit variance
   - ``test_scaler_reuse``: Tests reusing scaler on new data

**TestDataSplitting**
   - ``test_split_ratios``: Tests split respects given ratios
   - ``test_reproducibility``: Tests splits are reproducible with same seed
   - ``test_invalid_ratios``: Tests error for invalid ratios

Model Tests (test_model.py)
---------------------------

Tests for the ``NaiveMLP`` model.

**TestNaiveMLP**
   - ``test_default_initialization``: Tests default parameters
   - ``test_custom_hidden_dims``: Tests custom hidden layer dimensions
   - ``test_activation_relu``: Tests ReLU activation
   - ``test_activation_tanh``: Tests Tanh activation
   - ``test_activation_sigmoid``: Tests Sigmoid activation
   - ``test_invalid_activation``: Tests error for invalid activation
   - ``test_dropout_applied``: Tests dropout layers are added
   - ``test_no_dropout_when_zero``: Tests no dropout when rate is 0

**TestForwardPass**
   - ``test_forward_shape``: Tests output shape (batch_size, 1)
   - ``test_forward_single_sample``: Tests single sample forward pass
   - ``test_forward_large_batch``: Tests large batch forward pass
   - ``test_forward_deterministic``: Tests deterministic output in eval mode

**TestPredict**
   - ``test_predict_shape``: Tests squeezed output shape
   - ``test_predict_single_sample``: Tests single sample prediction
   - ``test_predict_no_grad``: Tests gradients are disabled

**TestModelProperties**
   - ``test_get_num_parameters``: Tests parameter counting
   - ``test_get_num_parameters_larger``: Tests counting for larger model
   - ``test_repr``: Tests string representation

**TestCreateModel**
   - ``test_create_with_full_config``: Tests factory with full config
   - ``test_create_with_minimal_config``: Tests factory with defaults
   - ``test_create_with_partial_config``: Tests factory with partial config

**TestGradientFlow**
   - ``test_gradients_flow``: Tests gradients flow through network
   - ``test_training_step``: Tests weights update after training step

Trainer Tests (test_trainer.py)
-------------------------------

Tests for the ``NNTrainer`` class.

**TestTrainerInitialization**
   - ``test_init_with_model_class``: Tests initialization with model class
   - ``test_init_with_factory_function``: Tests initialization with factory
   - ``test_default_training_config``: Tests default configuration
   - ``test_custom_training_config``: Tests custom configuration
   - ``test_device_auto_detection``: Tests automatic device detection
   - ``test_explicit_device``: Tests explicit device setting
   - ``test_checkpoint_dir_created``: Tests checkpoint directory creation

**TestTraining**
   - ``test_fit_runs``: Tests fit completes without error
   - ``test_fit_with_validation``: Tests training with validation data
   - ``test_loss_decreases``: Tests training loss decreases
   - ``test_early_stopping``: Tests early stopping triggers
   - ``test_history_tracking``: Tests training history is tracked

**TestEvaluation**
   - ``test_evaluate_returns_metrics``: Tests metrics are returned
   - ``test_rmse_is_sqrt_mse``: Tests RMSE equals sqrt(MSE)
   - ``test_metrics_are_positive``: Tests all metrics are non-negative

**TestPrediction**
   - ``test_predict_shape``: Tests prediction output shape
   - ``test_predict_single_sample``: Tests single sample prediction
   - ``test_predict_is_deterministic``: Tests predictions are deterministic

**TestCheckpointing**
   - ``test_save_and_load_model``: Tests model saving and loading
   - ``test_best_checkpoint_saved``: Tests best checkpoint is saved
