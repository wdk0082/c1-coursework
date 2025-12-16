Running Tests
=============

This guide covers how to run the test suite and interpret the results.

Quick Start
-----------

The fastest way to run all tests:

.. code-block:: bash

   ./scripts/run_test.sh

This script automatically:

- Activates the Python virtual environment
- Changes to the backend directory
- Runs pytest with verbose output

Basic Usage
-----------

Running All Tests
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Using the convenience script
   ./scripts/run_test.sh

   # Or manually
   source .venv/bin/activate
   cd backend
   pytest

Running with Verbose Output
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pytest -v

   # Or using the script
   ./scripts/run_test.sh -v

Running Specific Test Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Test API endpoints
   pytest test/test_api.py

   # Test data loading
   pytest test/test_data.py

   # Test model
   pytest test/test_model.py

   # Test trainer
   pytest test/test_trainer.py

Running Specific Test Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Test only health endpoint
   pytest test/test_api.py::TestHealthEndpoint

   # Test only upload endpoint
   pytest test/test_api.py::TestUploadEndpoint

   # Test only train endpoint
   pytest test/test_api.py::TestTrainEndpoint

Running Specific Test Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run a single test
   pytest test/test_api.py::TestHealthEndpoint::test_health_check

   # Run by keyword
   pytest -k "health"

Coverage Reports
----------------

Generate a coverage report to see which code is tested:

Basic Coverage
^^^^^^^^^^^^^^

.. code-block:: bash

   pytest --cov=fivedreg

   # Using the script
   ./scripts/run_test.sh --cov=fivedreg

Detailed Coverage Report
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Show missing lines
   pytest --cov=fivedreg --cov-report=term-missing

   # Generate HTML report
   pytest --cov=fivedreg --cov-report=html

   # View HTML report
   open htmlcov/index.html

Coverage Summary Example
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   ---------- coverage: platform darwin, python 3.11.0 ----------
   Name                              Stmts   Miss  Cover   Missing
   ---------------------------------------------------------------
   fivedreg/__init__.py                  2      0   100%
   fivedreg/data/__init__.py             0      0   100%
   fivedreg/data/load_data.py          115      5    96%   234-238
   fivedreg/model/__init__.py            0      0   100%
   fivedreg/model/naive_nn.py           65      2    97%   95-96
   fivedreg/trainer/__init__.py          0      0   100%
   fivedreg/trainer/nn_trainer.py      145      8    94%   312-320
   ---------------------------------------------------------------
   TOTAL                               327     15    95%
