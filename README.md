# 5D Regression

A full-stack application for training and deploying neural network models that perform 5-dimensional to 1-dimensional regression. The project consists of a FastAPI backend for model training and inference, and a Next.js frontend for interactive dataset management and model exploration. It includes a comprehensive test suite with coverage reporting and performance profiling tools for benchmarking model training across different dataset sizes and configurations.

## Table of Contents

- [5D Regression](#5d-regression)
  - [Table of Contents](#table-of-contents)
  - [1. Project Description](#1-project-description)
    - [Architecture](#architecture)
    - [Tech Stack](#tech-stack)
  - [2. Quick Start Guide](#2-quick-start-guide)
    - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
    - [Option 2: Local Deployment](#option-2-local-deployment)
    - [Basic Workflow](#basic-workflow)
    - [Creating a Sample Dataset](#creating-a-sample-dataset)
  - [3. Local Deployment](#3-local-deployment)
    - [Prerequisites](#prerequisites)
    - [Setup (for MacOS)](#setup-for-macos)
    - [Running the Application](#running-the-application)
  - [4. Test Suite (Local Deployment Required)](#4-test-suite-local-deployment-required)
    - [Running Tests](#running-tests)
    - [Test Structure](#test-structure)
  - [5. License](#5-license)

## 1. Project Description

This application provides an end-to-end machine learning workflow for regression tasks:

- **Dataset Management**: Upload `.pkl` datasets containing 5D input features and 1D targets
- **Model Training**: Train configurable neural networks with customizable architecture (hidden layers, activation functions, dropout) and training parameters (learning rate, batch size, epochs, early stopping)
- **Predictions**: Make single or batch predictions using trained models
- **Data Processing**: Built-in support for data standardization and missing value handling strategies

### Architecture

```
c1-coursework/
├── backend/                    # Core ML library and FastAPI backend
│   ├── api.py                  # API endpoints
│   ├── fivedreg/               # Core ML library
│   │   ├── data/               
│   │   ├── model/              
│   │   └── trainer/            
│   ├── test/                   # Test suite
│   └── docs/                   # Sphinx documentation
├── frontend/                   # Next.js frontend
│   └── src/
│       ├── app/                
│       ├── components/         
│       └── lib/api/            
├── profiling/                  # Performance profiling
│   ├── data/                   
│   └── results/                
├── scripts/                    # Utility scripts
└── docker-compose.yml          # Docker orchestration
```

### Tech Stack

**Backend**: Python 3.9+, FastAPI, PyTorch, NumPy

**Frontend**: Next.js 14, TypeScript, Tailwind CSS, Axios

## 2. Quick Start Guide

### Option 1: Docker (Recommended)

Just clone the repository and run:

```bash
git clone <repository-url>
cd c1-coursework
./scripts/docker_start.sh
```

To stop the application:
```bash
./scripts/docker_stop.sh
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

### Option 2: Local Deployment

See [Local Deployment](#3-local-deployment) for installation and setup instructions.

### Basic Workflow

1. **Upload a dataset**: Navigate to http://localhost:3000/upload and upload a `.pkl` file containing:
   - `X`: NumPy array of shape `(n_samples, 5)` - 5D input features
   - `y`: NumPy array of shape `(n_samples,)` - 1D output targets

2. **Train a model**: Go to http://localhost:3000/train, select your dataset, configure the model architecture and training parameters, then start training.

3. **Make predictions**: Visit http://localhost:3000/predict, select a trained model, and enter 5D input vectors to get predictions.

### Creating a Sample Dataset

```python
import numpy as np
import pickle

# Generate sample data
X = np.random.randn(1000, 5)
y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]

# Save as .pkl
with open('sample_dataset.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)
```

## 3. Local Deployment

### Prerequisites

- Python 3.9 or higher
- Node.js 18.0 or higher
- pip and npm package managers

### Setup (for MacOS)

**Quick Setup (Recommended)**:
```bash
git clone <repository-url>
cd c1-coursework
./scripts/simple_setup.sh
```

**Manual Setup**:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd c1-coursework
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Backend dependencies (including docs and test)
   cd backend
   pip install -e ".[docs,test]"
   cd ..

   # Frontend dependencies
   cd frontend
   npm install
   cd ..
   ```

4. **Build the documentation**:
   ```bash
   ./scripts/build_docs.sh
   # Documentation will be available at backend/docs/_build/html/index.html
   ```

5. **Configure frontend environment**:
   ```bash
   cd frontend
   cp .env.example .env.local
   # Edit .env.local if backend runs on a different URL
   ```

### Running the Application

```bash
# Start both backend and frontend
./scripts/local_start.sh

# Stop the application
./scripts/local_stop.sh
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

## 4. Test Suite (Local Deployment Required)

The backend includes a comprehensive test suite using pytest.

### Running Tests

**Quick Start (Recommended)**:
```bash
./scripts/run_test.sh
```

This script automatically activates the virtual environment and runs all tests with verbose output. You can also pass additional pytest arguments:
```bash
./scripts/run_test.sh --cov=fivedreg    # Run with coverage
./scripts/run_test.sh test/test_api.py  # Run specific test file
```

**Manual Testing**:

First, activate the virtual environment:
```bash
source .venv/bin/activate
cd backend
```

Then run pytest commands:
```bash
pytest                # Run all tests
pytest -v             # Run with verbose output
```

**Run specific test files**:
```bash
pytest test/test_api.py      # API endpoint tests
pytest test/test_model.py    # Model tests
pytest test/test_data.py     # Data loading tests
pytest test/test_trainer.py  # Trainer tests
```

**Run with coverage report**:
```bash
pytest --cov=fivedreg --cov-report=term-missing
```

**Run a specific test class or function**:
```bash
pytest test/test_api.py::TestHealthEndpoint
pytest test/test_api.py::TestHealthEndpoint::test_health_check
```

### Test Structure

```
backend/test/
├── conftest.py       # Shared fixtures (sample data, configurations)
├── test_api.py       # FastAPI endpoint tests
├── test_data.py      # Data loading and preprocessing tests
├── test_model.py     # Neural network model tests
└── test_trainer.py   # Training loop tests
```

## 5. License

MIT License - see [LICENSE](LICENSE) for details.