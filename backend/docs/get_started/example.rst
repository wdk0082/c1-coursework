An Example For Using fivedreg in Python
==========================================

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
