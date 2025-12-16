fivedreg Package Guide
======================

This guide provides comprehensive examples for using the ``fivedreg`` Python package to load data, train models, and make predictions.

Basic Workflow
--------------

A typical workflow involves loading data, creating a trainer, training the model, and making predictions:

.. code-block:: python

   from fivedreg.data.load_data import load_data
   from fivedreg.model.naive_nn import NaiveMLP
   from fivedreg.trainer.nn_trainer import NNTrainer

   # 1. Load and preprocess data
   data = load_data(
       "dataset.pkl",
       standardize=True,
       split_ratios=(0.7, 0.15, 0.15)
   )

   # 2. Create trainer with model and training configuration
   trainer = NNTrainer(
       model_class=NaiveMLP,
       model_config={"hidden_dims": [64, 32], "dropout": 0.1},
       training_config={"epochs": 100, "learning_rate": 1e-3}
   )

   # 3. Train the model
   result = trainer.fit(
       X_train=data["X_train"], y_train=data["y_train"],
       X_val=data["X_val"], y_val=data["y_val"]
   )

   # 4. Evaluate and predict
   metrics = trainer.evaluate(data["X_test"], data["y_test"])
   print(f"Test MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}")

   predictions = trainer.predict(data["X_test"])

Loading Data
------------

Creating a Dataset
^^^^^^^^^^^^^^^^^^

The package expects data with 5D input features and 1D targets:

.. code-block:: python

   import numpy as np
   import pickle

   # Generate synthetic data
   np.random.seed(42)
   n_samples = 1000

   X = np.random.randn(n_samples, 5)
   y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 - X[:, 2] * 0.5 +
        np.random.randn(n_samples) * 0.2)

   # Save as pickle
   with open("dataset.pkl", "wb") as f:
       pickle.dump({"X": X, "y": y}, f)

   # Or save as npz
   np.savez("dataset.npz", X=X, y=y)

Loading with Different Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from fivedreg.data.load_data import load_data

   # Basic loading
   data = load_data("dataset.pkl")
   X, y = data["X"], data["y"]

   # With train/val/test split
   data = load_data("dataset.pkl", split_ratios=(0.7, 0.15, 0.15))
   X_train, y_train = data["X_train"], data["y_train"]
   X_val, y_val = data["X_val"], data["y_val"]
   X_test, y_test = data["X_test"], data["y_test"]

   # With standardization (recommended)
   data = load_data("dataset.pkl", standardize=True, split_ratios=(0.7, 0.15, 0.15))
   scaler = data["scaler"]  # Contains mean and std for inference

Handling Missing Values
^^^^^^^^^^^^^^^^^^^^^^^

Five strategies are available for handling NaN values:

.. code-block:: python

   # Remove rows with NaN (default)
   data = load_data("data.pkl", missing_strategy="ignore")

   # Fill with column mean
   data = load_data("data.pkl", missing_strategy="mean")

   # Fill with column median
   data = load_data("data.pkl", missing_strategy="median")

   # Fill with zeros
   data = load_data("data.pkl", missing_strategy="zero")

   # Forward fill
   data = load_data("data.pkl", missing_strategy="forward_fill")

Configuring Models
------------------

Model Architecture
^^^^^^^^^^^^^^^^^^

The ``NaiveMLP`` model accepts these configuration options:

.. code-block:: python

   from fivedreg.model.naive_nn import NaiveMLP

   # Default architecture: [64, 32] hidden layers
   model = NaiveMLP()

   # Custom architecture
   model = NaiveMLP(
       hidden_dims=[128, 64, 32],  # Three hidden layers
       dropout=0.2,                 # 20% dropout
       activation="relu"            # relu, tanh, or sigmoid
   )

   print(f"Parameters: {model.get_num_parameters():,}")

Architecture Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

- **Small datasets (<1000 samples)**: Use shallow networks like ``[32, 16]``
- **Medium datasets (1000-10000)**: Try ``[64, 32]`` or ``[128, 64]``
- **Large datasets (>10000)**: Can use deeper networks like ``[256, 128, 64]``
- **Overfitting**: Add dropout (0.1-0.3) or reduce network size

Training Models
---------------

Basic Training
^^^^^^^^^^^^^^

.. code-block:: python

   from fivedreg.trainer.nn_trainer import NNTrainer
   from fivedreg.model.naive_nn import NaiveMLP

   trainer = NNTrainer(
       model_class=NaiveMLP,
       model_config={"hidden_dims": [64, 32]},
       training_config={
           "learning_rate": 1e-3,
           "batch_size": 32,
           "epochs": 100,
           "weight_decay": 0.0
       }
   )

   result = trainer.fit(X_train, y_train, X_val, y_val, verbose=True)

Training Results
^^^^^^^^^^^^^^^^

The ``fit()`` method returns a dictionary with training information:

.. code-block:: python

   result = trainer.fit(X_train, y_train, X_val, y_val)

   print(f"Best epoch: {result['best_epoch']}")
   print(f"Best validation loss: {result['best_val_loss']:.4f}")
   print(f"Training time: {result['training_time_seconds']:.2f}s")
   print(f"Memory usage: {result['training_memory_mb']:.2f} MB")

   # Access training history
   train_losses = result["history"]["train_loss"]
   val_losses = result["history"]["val_loss"]

Evaluation and Prediction
-------------------------

Evaluating Models
^^^^^^^^^^^^^^^^^

.. code-block:: python

   metrics = trainer.evaluate(X_test, y_test)

   print(f"MSE:  {metrics['mse']:.4f}")
   print(f"RMSE: {metrics['rmse']:.4f}")
   print(f"MAE:  {metrics['mae']:.4f}")
   print(f"R2:   {metrics['r2']:.4f}")

Making Predictions
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Basic prediction
   predictions = trainer.predict(X_new)

   # With memory tracking
   predictions, memory_mb = trainer.predict(X_new, track_memory=True)

Saving and Loading Models
-------------------------

.. code-block:: python

   # Save after training
   trainer.save_model("model.pt")

   # Load later
   new_trainer = NNTrainer(
       model_class=NaiveMLP,
       model_config={"hidden_dims": [64, 32]},  # Must match saved model
       training_config={}
   )
   new_trainer.load_model("model.pt")

   # Use loaded model
   predictions = new_trainer.predict(X_new)

Complete Example
----------------

Here's a complete end-to-end example:

.. code-block:: python

   import numpy as np
   import pickle
   from fivedreg.data.load_data import load_data
   from fivedreg.model.naive_nn import NaiveMLP
   from fivedreg.trainer.nn_trainer import NNTrainer

   # Create sample dataset
   np.random.seed(42)
   X = np.random.randn(2000, 5)
   y = X[:, 0] * 2 + X[:, 1] - X[:, 2] * 0.5 + np.random.randn(2000) * 0.1

   with open("example.pkl", "wb") as f:
       pickle.dump({"X": X, "y": y}, f)

   # Load and preprocess
   data = load_data(
       "example.pkl",
       standardize=True,
       split_ratios=(0.7, 0.15, 0.15)
   )

   # Train model
   trainer = NNTrainer(
       model_class=NaiveMLP,
       model_config={"hidden_dims": [64, 32], "dropout": 0.1},
       training_config={"epochs": 50, "learning_rate": 1e-3}
   )

   result = trainer.fit(
       data["X_train"], data["y_train"],
       data["X_val"], data["y_val"],
       verbose=True
   )

   # Evaluate
   metrics = trainer.evaluate(data["X_test"], data["y_test"])
   print(f"\nTest Results: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}")

   # Save model
   trainer.save_model("trained_model.pt")

   # Make predictions on new data
   X_new = np.array([[0.5, -0.3, 0.8, 0.1, -0.2]])
   X_scaled = (X_new - data["scaler"]["mean"]) / data["scaler"]["std"]
   prediction = trainer.predict(X_scaled)
   print(f"Prediction for new input: {prediction[0]:.4f}")
