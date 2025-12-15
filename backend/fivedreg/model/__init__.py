"""
Model Module
============

This module provides neural network model definitions for regression tasks.

Classes
-------
NaiveMLP
    Simple Multi-Layer Perceptron for regression.

Functions
---------
create_model
    Factory function to create models from configuration dictionaries.
"""

from .naive_nn import NaiveMLP, create_model

__all__ = ["NaiveMLP", "create_model"]
