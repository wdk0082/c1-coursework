Data Module
===========

Data loading and preprocessing utilities.

Overview
--------

The ``load_data`` function is the primary interface for loading datasets. It handles:

- Multiple file formats (``.npz``, ``.pkl``, ``.pickle``)
- Various data key naming conventions (``X``/``y``, ``inputs``/``outputs``, ``features``/``targets``)
- Missing value handling with multiple strategies
- Feature standardization
- Train/validation/test splitting

API Reference
-------------

.. automodule:: fivedreg.data.load_data
   :members:
   :undoc-members:
   :show-inheritance:
