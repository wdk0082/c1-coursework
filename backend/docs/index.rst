5D Regression Documentation
===========================

Welcome to the 5D Regression documentation. This application provides an end-to-end machine learning workflow for training and deploying neural network models that perform 5-dimensional to 1-dimensional regression.

Overview
--------

The 5D Regression platform consists of:

- **Backend API**: FastAPI-based REST API for dataset management, model training, and inference
- **fivedreg Package**: Core Python library for data loading, neural network models, and training utilities
- **Frontend UI**: Next.js web application for interactive model training and predictions

Key Features
------------

- **Dataset Management**: Upload and validate ``.npz`` and ``.pkl`` datasets
- **Configurable Models**: Multi-layer perceptrons with customizable architecture
- **Training Pipeline**: Automated training with validation, checkpointing, and metrics
- **Missing Value Handling**: Five strategies for handling incomplete data
- **Feature Standardization**: Automatic feature scaling with train-set fitting
- **REST API**: Complete API for programmatic access

.. toctree::
   :maxdepth: 1
   :caption: 1 Get Started

   get_started/installation
   get_started/quickstart
   get_started/example

.. toctree::
   :maxdepth: 1
   :caption: 2 User Guide

   user_guide/fivedreg_guide
   user_guide/api_endpoints
   user_guide/frontend

.. toctree::
   :maxdepth: 1
   :caption: 3 API Reference

   api_reference/data
   api_reference/model
   api_reference/trainer

.. toctree::
   :maxdepth: 1
   :caption: 4 Test Suite

   test_suite/running_tests
   test_suite/test_structure

