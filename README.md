# 5D Regression

A full-stack application for training and deploying neural network models that perform 5-dimensional to 1-dimensional regression. The project consists of a FastAPI backend for model training and inference, and a Next.js frontend for interactive dataset management and model exploration.

## Project Description

This application provides an end-to-end machine learning workflow for regression tasks:

- **Dataset Management**: Upload `.npz` datasets containing 5D input features and 1D targets
- **Model Training**: Train configurable neural networks with customizable architecture (hidden layers, activation functions, dropout) and training parameters (learning rate, batch size, epochs, early stopping)
- **Predictions**: Make single or batch predictions using trained models
- **Data Processing**: Built-in support for data standardization and missing value handling strategies

### Architecture

```
c1-coursework/
├── backend/                    # FastAPI backend
│   ├── api.py                 # API endpoints
│   ├── fivedreg/              # Core ML library
│   │   ├── data/              # Data loading utilities
│   │   ├── model/             # Neural network models
│   │   └── trainer/           # Training utilities
│   ├── test/                  # Test suite
│   └── docs/                  # Sphinx documentation
├── frontend/                   # Next.js frontend
│   └── src/
│       ├── app/               # Pages (dashboard, upload, train, predict, models)
│       ├── components/        # React components
│       └── lib/api/           # API client
├── scripts/                    # Utility scripts
└── docker-compose.yml          # Docker orchestration
```

### Tech Stack

**Backend**: Python 3.9+, FastAPI, PyTorch, NumPy

**Frontend**: Next.js 14, TypeScript, Tailwind CSS, Axios

## Installation Guide

### Prerequisites

- Python 3.9 or higher
- Node.js 18.0 or higher
- pip and npm package managers

### Setup

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

3. **Install backend dependencies**:
   ```bash
   cd backend
   pip install -e .
   ```

4. **Install documentation dependencies** (optional):
   ```bash
   pip install -e ".[docs]"
   ```

5. **Build the documentation** (optional):
   ```bash
   cd ..
   ./scripts/build_docs.sh
   # Documentation will be available at backend/docs/_build/html/index.html
   ```

6. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

7. **Configure frontend environment**:
   ```bash
   cp .env.example .env.local
   # Edit .env.local if backend runs on a different URL
   ```

## Quick Start Guide

### Option 1: Run with Docker (Recommended)

The easiest way to run the full application:

```bash
# Start both backend and frontend
./scripts/docker_start.sh

# Stop the application
./scripts/docker_stop.sh
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs

### Option 2: Run Manually

**Terminal 1 - Start the backend**:
```bash
source .venv/bin/activate
cd backend
python api.py
```

**Terminal 2 - Start the frontend**:
```bash
cd frontend
npm run dev
```

### Basic Workflow

1. **Upload a dataset**: Navigate to http://localhost:3000/upload and upload a `.npz` file containing:
   - `X`: NumPy array of shape `(n_samples, 5)` - 5D input features
   - `y`: NumPy array of shape `(n_samples,)` - 1D output targets

2. **Train a model**: Go to http://localhost:3000/train, select your dataset, configure the model architecture and training parameters, then start training.

3. **Make predictions**: Visit http://localhost:3000/predict, select a trained model, and enter 5D input vectors to get predictions.

### Creating a Sample Dataset

```python
import numpy as np

# Generate sample data
X = np.random.randn(1000, 5)
y = X[:, 0] * 2 + X[:, 1] * 3 - X[:, 2]

# Save as .npz
np.savez('sample_dataset.npz', X=X, y=y)
```

## Test Suite Guide

The backend includes a comprehensive test suite using pytest.

### Install Test Dependencies

```bash
source .venv/bin/activate
pip install pytest pytest-cov httpx
```

### Running Tests

**Run all tests**:
```bash
cd backend
pytest
```

**Run with verbose output**:
```bash
pytest -v
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

## License

MIT License - see [LICENSE](LICENSE) for details.

Academic project for Cambridge C1 coursework.
