Installation
============

This guide covers the installation process for the 5D Regression application. You can run the application either with Docker (recommended for quick setup) or locally for development.

Prerequisites
-------------

Before installing, ensure you have the following prerequisites based on your deployment method.

For Docker Deployment
^^^^^^^^^^^^^^^^^^^^^

- **Docker**: Version 20.0 or higher
- **Docker Compose**: Version 2.0 or higher (usually included with Docker Desktop)

**Installing Docker:**

- **macOS**: Download and install `Docker Desktop for Mac <https://docs.docker.com/desktop/install/mac-install/>`_
- **Windows**: Download and install `Docker Desktop for Windows <https://docs.docker.com/desktop/install/windows-install/>`_
- **Linux**: Follow the `Docker Engine installation guide <https://docs.docker.com/engine/install/>`_

Verify your Docker installation:

.. code-block:: bash

   docker --version
   docker compose version

For Local Deployment
^^^^^^^^^^^^^^^^^^^^

- **Python**: Version 3.9 or higher
- **Node.js**: Version 18.0 or higher
- **pip**: Python package manager (usually included with Python)
- **npm**: Node.js package manager (usually included with Node.js)
- **Git**: For cloning the repository

Docker Installation (Recommended)
---------------------------------

The Docker installation provides a containerized environment with all dependencies pre-configured.

1. **Clone the repository**:

   .. code-block:: bash

      git clone <repository-url>
      cd zw499

2. **Start the application**:

   .. code-block:: bash

      ./scripts/docker_start.sh

3. **Access the application**:

   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

4. **Stop the application**:

   .. code-block:: bash

      ./scripts/docker_stop.sh

That's it. Docker handles all the dependency installation automatically.

Local Installation
------------------

For development or if you prefer running without Docker, follow these steps.

Quick Setup (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

Use the provided setup script for automatic installation:

.. code-block:: bash

   git clone <repository-url>
   cd zw499
   ./scripts/simple_setup.sh

This script automatically:

- Creates a Python virtual environment (``.venv``)
- Installs all backend dependencies (including test and docs packages)
- Installs frontend dependencies
- Configures the frontend environment (``.env.local``)
- Builds the Sphinx documentation

Manual Setup
^^^^^^^^^^^^

If you prefer manual installation:

1. **Clone the repository**:

   .. code-block:: bash

      git clone <repository-url>
      cd zw499

2. **Create and activate a Python virtual environment**:

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. **Install backend dependencies**:

   .. code-block:: bash

      cd backend
      pip install -e ".[docs,test]"
      cd ..

   This installs:

   - Core dependencies: PyTorch, FastAPI, NumPy
   - Documentation dependencies: Sphinx, sphinx-rtd-theme
   - Test dependencies: pytest, pytest-cov

4. **Install frontend dependencies**:

   .. code-block:: bash

      cd frontend
      npm install
      cd ..

5. **Configure frontend environment**:

   .. code-block:: bash

      cd frontend
      cp .env.example .env.local
      cd ..

   Edit ``.env.local`` if your backend runs on a different URL.

6. **Build the documentation** (optional):

   .. code-block:: bash

      ./scripts/build_docs.sh

Next Steps
----------

Once installation is complete, proceed to the :doc:`/get_started/quickstart` guide to learn how to use the application.
