# Astronomical Data Statistical Estimator

A specialized Python data analysis engine built to process, clean, and model space and astronomical datasets. This pipeline utilizes custom validation algorithms and matrix mechanics to extract statistical meaning from noisy telemetry and spatial data.

## Features
*   **Custom Parsing Algorithms:** Engineered `while`-loop data ingesters designed to handle large, corrupted datasets without execution errors or tracebacks.
*   **Edge-Case Handlers:** Specifically built to trap and resolve edge-case inputs (e.g., telemetry dropouts at Input 1 and sensor saturation spikes at Input 7).
*   **Statistical Estimation:** Calculates joint probability density functions across astronomical variables using 2D histograms and SciPy distributions.
*   **Spatial Modeling:** Applies matrix mechanics, calculating spatial gradients, Hessian determinants, and Singular Value Decompositions (SVD) to model mathematical transformations accurately.

## Tech Stack
*   Python 3.x
*   `NumPy`
*   `Pandas`
*   `SciPy`

## Execution
Run the main estimator engine to parse the raw data and generate the transformed spatial models:
```bash
python astro_estimator.py
