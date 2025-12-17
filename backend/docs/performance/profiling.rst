Performance Profiling
=====================

This document presents profiling results for the 5D Regression model across various synthetic datasets. The presented results are drawn from one run of the profiling suite. We also provide scripts on how to run the profiling benchmarks yourself at the end.

Overview
--------

The profiling suite benchmarks the ``NaiveMLP`` model on five different function types with varying dataset sizes and input scales. The goal is to evaluate:

- **Training time**: How long training takes as dataset size increases
- **Memory usage**: Training and inference memory consumption
- **Model accuracy**: MSE and R² metrics on train and test sets

Profiling Configuration
-----------------------

**Model Configuration**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Value
   * - Hidden Dimensions
     - [64, 32]
   * - Dropout
     - 0.0
   * - Activation
     - ReLU

**Training Configuration**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Value
   * - Learning Rate
     - 0.001
   * - Batch Size
     - 32
   * - Epochs
     - 100
   * - Weight Decay
     - 0.0

**Dataset Configuration**

- **Split Ratios**: 70% train, 15% validation, 15% test
- **Noise Std**: 0.1
- **Random Seed**: 42
- **Sizes**: 1K, 5K, 10K samples
- **Scales**: Small (-1, 1), Large (-10, 10)

Results by Function Type
------------------------

Linear Function
~~~~~~~~~~~~~~~

The linear function represents the simplest regression task: :math:`y = \sum_{i=1}^{5} w_i x_i + b`.

.. figure:: /_static/profiling/linear_20251216_112116.png
   :align: center
   :width: 100%

   Linear function profiling results.

.. list-table:: Linear Function - Performance Metrics
   :widths: 20 13 13 13 13 13 13
   :header-rows: 1

   * - Metric
     - 1K small
     - 5K small
     - 10K small
     - 1K large
     - 5K large
     - 10K large
   * - Training Time (s)
     - 1.50
     - 6.72
     - 13.80
     - 1.47
     - 6.82
     - 13.70
   * - Train Memory (MB)
     - 0.50
     - 0.36
     - 0.47
     - 0.38
     - 0.43
     - 0.60
   * - Train MSE
     - 0.0095
     - 0.0086
     - 0.0094
     - 0.2651
     - 0.0154
     - 0.0130
   * - Test MSE
     - 0.0337
     - 0.0118
     - 0.0109
     - 0.4398
     - 0.0174
     - 0.0136
   * - Train R²
     - 0.9985
     - 0.9987
     - 0.9986
     - 0.9996
     - 0.9999+
     - 0.9999+
   * - Test R²
     - 0.9955
     - 0.9982
     - 0.9983
     - 0.9994
     - 0.9999+
     - 0.9999+

**Analysis**: The model achieves excellent performance on linear data, with R² > 0.99 across all configurations. Training time scales linearly with dataset size. The large scale datasets show slightly higher MSE in absolute terms but maintain excellent R² scores.

Polynomial Function
~~~~~~~~~~~~~~~~~~~

The polynomial function tests the model on nonlinear relationships: :math:`y = \sum_{i=1}^{5} x_i^2 + \sum_{i<j} x_i x_j`.

.. figure:: /_static/profiling/polynomial_20251216_112116.png
   :align: center
   :width: 100%

   Polynomial function profiling results.

.. list-table:: Polynomial Function - Performance Metrics
   :widths: 20 13 13 13 13 13 13
   :header-rows: 1

   * - Metric
     - 1K small
     - 5K small
     - 10K small
     - 1K large
     - 5K large
     - 10K large
   * - Training Time (s)
     - 1.41
     - 6.75
     - 13.58
     - 1.38
     - 6.83
     - 13.71
   * - Train Memory (MB)
     - 0.37
     - 0.36
     - 0.50
     - 0.27
     - 0.50
     - 0.63
   * - Train MSE
     - 0.0091
     - 0.0089
     - 0.0101
     - 22808.82
     - 363.46
     - 98.59
   * - Test MSE
     - 0.0352
     - 0.0148
     - 0.0130
     - 25774.44
     - 420.99
     - 109.58
   * - Train R²
     - 0.9787
     - 0.9793
     - 0.9765
     - 0.8411
     - 0.9974
     - 0.9993
   * - Test R²
     - 0.9176
     - 0.9666
     - 0.9710
     - 0.8626
     - 0.9968
     - 0.9993

**Analysis**: Polynomial functions are more challenging. Small scale data achieves R² ~0.97. Large scale data with 1K samples shows lower R² (0.86), but performance improves significantly with more training data (R² = 0.9993 at 10K). This demonstrates the importance of dataset size for complex functions.

Sinusoidal Function
~~~~~~~~~~~~~~~~~~~

The sinusoidal function tests the model on periodic patterns: :math:`y = \sum_{i=1}^{5} \sin(x_i)`.

.. figure:: /_static/profiling/sin_20251216_112116.png
   :align: center
   :width: 100%

   Sinusoidal function profiling results.

.. list-table:: Sinusoidal Function - Performance Metrics
   :widths: 20 13 13 13 13 13 13
   :header-rows: 1

   * - Metric
     - 1K small
     - 5K small
     - 10K small
     - 1K large
     - 5K large
     - 10K large
   * - Training Time (s)
     - 1.41
     - 6.83
     - 13.78
     - 1.36
     - 6.66
     - 13.43
   * - Train Memory (MB)
     - 0.30
     - 0.49
     - 0.49
     - 0.07
     - 0.20
     - 0.32
   * - Train MSE
     - 0.1239
     - 0.0226
     - 0.0156
     - 2.0072
     - 1.9260
     - 1.9644
   * - Test MSE
     - 0.3305
     - 0.0310
     - 0.0209
     - 2.1373
     - 2.0162
     - 2.0188
   * - Train R²
     - 0.9250
     - 0.9847
     - 0.9897
     - 0.0100
     - 0.0335
     - 0.0157
   * - Test R²
     - 0.7598
     - 0.9787
     - 0.9854
     - -0.1069
     - -0.0212
     - -0.0101

**Analysis**: Sinusoidal functions show interesting behavior. On small scale data (-1, 1), the model learns well with R² reaching 0.98 at 10K samples. However, on large scale data (-10, 10), the model fails completely (R² near 0 or negative). This is because the neural network cannot approximate the high-frequency oscillations of sin(x) over a wide range with this architecture.

Exponential Function
~~~~~~~~~~~~~~~~~~~~

The exponential function tests rapid growth patterns: :math:`y = \exp(\sum_{i=1}^{5} x_i / 5)`.

.. figure:: /_static/profiling/expo_20251216_112116.png
   :align: center
   :width: 100%

   Exponential function profiling results.

.. list-table:: Exponential Function - Performance Metrics
   :widths: 20 13 13 13 13 13 13
   :header-rows: 1

   * - Metric
     - 1K small
     - 5K small
     - 10K small
     - 1K large
     - 5K large
     - 10K large
   * - Training Time (s)
     - 1.44
     - 6.74
     - 13.60
     - 1.45
     - 6.84
     - 13.78
   * - Train Memory (MB)
     - 0.38
     - 0.34
     - 0.43
     - 0.39
     - 0.49
     - 0.63
   * - Train MSE
     - 0.0098
     - 0.0092
     - 0.0103
     - 6.28M
     - 0.94M
     - 75659.55
   * - Test MSE
     - 0.0228
     - 0.0125
     - 0.0117
     - 9.42M
     - 0.92M
     - 78778.20
   * - Train R²
     - 0.9906
     - 0.9911
     - 0.9898
     - 0.4234
     - 0.9178
     - 0.9933
   * - Test R²
     - 0.9782
     - 0.9881
     - 0.9884
     - 0.4307
     - 0.9147
     - 0.9930

**Analysis**: Exponential functions behave similarly to polynomial. Small scale achieves R² ~0.99. Large scale performance improves dramatically with more data: from R² = 0.43 at 1K to R² = 0.99 at 10K samples. The absolute MSE values are high for large scale due to the exponential growth, but relative performance (R²) is excellent.

Mixed Function
~~~~~~~~~~~~~~

The mixed function combines multiple patterns: :math:`y = x_1 + x_2^2 + \sin(x_3) + \exp(x_4/5) + x_5 \cdot x_1`.

.. figure:: /_static/profiling/mixed_20251216_112116.png
   :align: center
   :width: 100%

   Mixed function profiling results.

.. list-table:: Mixed Function - Performance Metrics
   :widths: 20 13 13 13 13 13 13
   :header-rows: 1

   * - Metric
     - 1K small
     - 5K small
     - 10K small
     - 1K large
     - 5K large
     - 10K large
   * - Training Time (s)
     - 1.40
     - 6.76
     - 13.64
     - 1.47
     - 6.77
     - 13.69
   * - Train Memory (MB)
     - 0.39
     - 0.37
     - 0.47
     - 0.39
     - 0.50
     - 0.63
   * - Train MSE
     - 0.0143
     - 0.0103
     - 0.0105
     - 34.99
     - 4.50
     - 1.56
   * - Test MSE
     - 0.0539
     - 0.0163
     - 0.0134
     - 55.21
     - 5.25
     - 1.85
   * - Train R²
     - 0.9886
     - 0.9920
     - 0.9921
     - 0.9830
     - 0.9980
     - 0.9993
   * - Test R²
     - 0.9622
     - 0.9869
     - 0.9898
     - 0.9777
     - 0.9975
     - 0.9991

**Analysis**: The mixed function, despite combining multiple patterns, is well-approximated by the model. All configurations achieve R² > 0.96, with larger datasets showing R² > 0.99. This suggests the sinusoidal component has less impact when combined with other terms.

Summary
-------

**Key Findings**

1. **Training Time**: Scales linearly with dataset size (~1.4s for 1K, ~6.8s for 5K, ~13.7s for 10K samples at 100 epochs).

2. **Memory Usage**: Training memory is modest (0.3-0.6 MB). Inference memory is negligible (~0.002 MB).

3. **Linear Functions**: Achieve near-perfect R² (>0.99) across all configurations.

4. **Polynomial/Exponential**: Require larger datasets for accurate modeling on wide input ranges.

5. **Sinusoidal Functions**: The model fails on large-scale sinusoidal data due to architectural limitations in capturing high-frequency patterns.

6. **Mixed Functions**: Well-approximated despite containing multiple component types.

**Recommendations**

- For simple functions (linear, polynomial), small datasets (1K-5K) suffice
- For complex functions on wide input ranges, use 10K+ samples
- Consider deeper architectures or specialized activation functions for periodic patterns

Running the Profiling
---------------------

To run the profiling benchmarks:

.. code-block:: bash

   ./scripts/run_profiling.sh

This script:

1. Generates synthetic datasets for all function types, sizes, and scales
2. Trains models on each dataset
3. Records metrics
4. Generates visualization plots

Results are saved to ``profiling/results/``.
