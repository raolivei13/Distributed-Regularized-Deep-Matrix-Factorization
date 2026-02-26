# Distributed Regularized Deep Matrix Factorization

This repository is public and intended to share the **Python experiment code** used for a research paper on stability in distributed deep matrix factorization (DMF).

## Paper overview

This code supports a paper that studies the stability of **consensus-based gradient methods** for a non-convex distributed learning problem: deep matrix factorization (DMF).

In short, the paper extends classical (centralized) DMF analysis to a **distributed multi-agent setting** with consensus updates and weight-decay regularization. Its main contribution is a set of **tractable learning-rate conditions** for stable dynamics. These conditions depend on:

- the largest eigenvalue of the Hessian of the regularized loss, and
- the communication graph topology across agents.

The experiments in this repository are intended to numerically illustrate and validate those stability claims.

## What this code is about (short)

The `.py` files in this repository generate synthetic distributed DMF problems and run consensus-based gradient dynamics with regularization, mainly to:

- test learning-rate stability conditions,
- evaluate reconstruction and consensus behavior, and
- reproduce numerical experiments/plots for the paper.

## Files

- `main_dmf.py`: main experiment script and plot generation.
- `main_graph_tests.py`: experiments that vary graph connectivity.
- `dmf_consensus_w_2reg_.py`: consensus + gradient algorithm implementation.
- `helper_functions_dmf.py`: data generation and utility functions.
- `stability_metric.py`: additional/experimental stability-analysis code.

## Installation

1. Clone and enter the repository:

   ```bash
   git clone <YOUR_REPO_URL>
   cd Distributed-Regularized-Deep-Matrix-Factorization
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create plot output folder:

   ```bash
   mkdir -p saved_plots
   ```

## Run

```bash
python main_dmf.py
```

Optional:

```bash
python main_graph_tests.py
python stability_metric.py
```

## Notes

- Dependencies are listed in `requirements.txt`.
- Some scripts use `matplotlib.use("TkAgg")` and LaTeX plotting options; on headless systems you may need to switch backend to `Agg` and disable `text.usetex`.
