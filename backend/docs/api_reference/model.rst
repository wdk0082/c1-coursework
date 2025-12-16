Model Module
============

Neural network model definitions for 5D to 1D regression.

Overview
--------

The primary model is ``NaiveMLP``, a configurable Multi-Layer Perceptron that:

- Accepts 5-dimensional input
- Produces 1-dimensional output (regression)
- Supports configurable hidden layer dimensions
- Includes optional dropout regularization
- Offers multiple activation functions (relu, tanh, sigmoid)

API Reference
-------------

.. automodule:: fivedreg.model.naive_nn
   :members:
   :undoc-members:
   :show-inheritance:
