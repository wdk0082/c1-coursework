Quick Start
===========

This guide will help you get up and running with the 5D Regression application quickly.

Launching the Application
-------------------------

Docker Deployment
^^^^^^^^^^^^^^^^^

Start the application with a single command:

.. code-block:: bash

   ./scripts/docker_start.sh

This launches both the backend API and frontend in containers. Access:

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000

To stop:

.. code-block:: bash

   ./scripts/docker_stop.sh

Local Deployment
^^^^^^^^^^^^^^^^

Start the application locally:

.. code-block:: bash

   ./scripts/local_start.sh

To stop:

.. code-block:: bash

   ./scripts/local_stop.sh

Basic Workflow
--------------

1. **Upload a Dataset**

   Navigate to http://localhost:3000/upload and upload a ``.pkl`` or ``.npz`` file containing your 5D input features (``X``) and 1D targets (``y``).

2. **Train a Model**

   Go to http://localhost:3000/train, select your dataset, configure the model architecture and training parameters, then start training.

3. **Make Predictions**

   Visit http://localhost:3000/predict, select a trained model, and enter 5D input vectors to get predictions.

Creating a Sample Dataset
-------------------------

.. code-block:: python

   import numpy as np
   import pickle

   # Generate sample data
   X = np.random.randn(1000, 5)
   y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]

   # Save as .pkl
   with open('sample_dataset.pkl', 'wb') as f:
       pickle.dump({'X': X, 'y': y}, f)

Next Steps
----------

- See the :doc:`/user_guide/fivedreg_guide` for comprehensive usage examples
- Check the :doc:`/user_guide/api_endpoints` for REST API details
- Explore '3 API REFERENCE' for full API documentation
