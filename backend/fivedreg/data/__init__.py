"""
Data Module
===========

This module provides utilities for loading and preprocessing data
for 5D to 1D regression tasks.

Functions
---------
load_data
    Load and preprocess data from pickle or npz files.
"""

from .load_data import load_data

__all__ = ["load_data"]
