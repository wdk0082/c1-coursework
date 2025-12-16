Frontend Guide
==============

The web interface provides an intuitive way to interact with the 5D Regression platform. Access it at http://localhost:3000 when the application is running.

Pages Overview
--------------

Dashboard (/)
^^^^^^^^^^^^^

The home page displays:

- System health status and backend connection
- Current device (CPU/GPU)
- Count of loaded datasets and trained models
- Quick action links to other pages

Upload (/upload)
^^^^^^^^^^^^^^^^

Upload your dataset files here:

- Accepts ``.npz`` and ``.pkl`` files
- Files must contain ``X`` (n_samples x 5) and ``y`` (n_samples) arrays
- Shows upload progress and validation results

Train (/train)
^^^^^^^^^^^^^^

Configure and train models:

- Select an uploaded dataset
- Set model architecture (hidden layers, dropout, activation)
- Configure training parameters (learning rate, batch size, epochs)
- Choose data preprocessing options (standardization, missing value handling)
- View training progress and results

Predict (/predict)
^^^^^^^^^^^^^^^^^^

Make predictions with trained models:

- Select a trained model
- Enter 5D input values
- View prediction results

Datasets (/datasets)
^^^^^^^^^^^^^^^^^^^^

View all uploaded datasets:

- Dataset ID and filename
- Number of samples and data shape
- Upload timestamp

Models (/models)
^^^^^^^^^^^^^^^^

View all trained models:

- Model ID and associated dataset
- Training metrics (MSE, RMSE, MAE, R2)
- Training time and memory usage

Model Details (/models/{id})
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Detailed view of a specific model:

- Full architecture configuration
- Training parameters used
- Complete performance metrics

Technology Stack
----------------

The frontend is built with:

- **Next.js 14**: React-based framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first styling
- **Axios**: HTTP client for API communication
