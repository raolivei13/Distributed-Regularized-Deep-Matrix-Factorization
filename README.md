# Distributed Regularized Deep Matrix Factorization

This repository contains research code for **distributed deep matrix factorization (DMF)** with:

- multiple agents/nodes,
- consensus averaging over a communication graph (Metropolis-Hastings weights), and
- per-node \(\ell_2\) regularization.

The scripts generate synthetic data, run consensus-based optimization, and produce plots for reconstruction and consensus behavior.

## Repository structure

- `main_dmf.py`  
  Main experiment script. Builds synthetic problem instances, computes step sizes from Hessian-based sharpness bounds, runs distributed optimization, and saves plots.
- `dmf_consensus_w_2reg_.py`  
  Core algorithm implementation: node-wise losses/gradients, consensus + gradient updates, and Hessian-eigenvalue tracking via Hessian-vector products.
- `helper_functions_dmf.py`  
  Utility functions for data generation, graph weights, parameter stacking/unstacking, and consensus metrics.
- `main_graph_tests.py`  
  Experiment that varies graph connection probability and compares optimization/consensus behavior.
- `stability_metric.py`  
  Work-in-progress script for stability/spectral-radius analysis.

## Requirements

- Python **3.9+** (tested with standard CPython workflow)
- pip

Python dependencies are listed in `requirements.txt`.

## Installation (step-by-step)

1. Clone the repository:

   ```bash
   git clone <YOUR_REPO_URL>
   cd Distributed-Regularized-Deep-Matrix-Factorization
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create output folder for figures (some scripts save into this path):

   ```bash
   mkdir -p saved_plots
   ```

## Running the code

### Main experiment

```bash
python main_dmf.py
```

Expected outputs:
- terminal logs of optimization iterations,
- figures saved in `saved_plots/` such as:
  - `sharpness_plot.png`
  - `normalized_errors.png`
  - `consensus_errors.png`

### Graph connectivity experiment

```bash
python main_graph_tests.py
```

### Stability script (experimental)

```bash
python stability_metric.py
```

## Notes and troubleshooting

1. **Matplotlib backend (`TkAgg`)** is hard-coded in scripts. If your machine does not have Tk installed (common in headless servers), change:

   ```python
   matplotlib.use("TkAgg")
   ```

   to:

   ```python
   matplotlib.use("Agg")
   ```

2. **LaTeX rendering is enabled** (`text.usetex = True`). For publication-quality labels this is useful, but it requires a local LaTeX installation.  
   If LaTeX is missing, set `text.usetex` to `False` in the script you run.

3. Scripts are currently designed around synthetic-data experiments and fixed hyperparameters defined near the top of each script.

## Reproducibility tips

- Keep `numpy` random seeds fixed where provided (`np.random.RandomState(0)` is used for graph generation in some scripts).
- Start with smaller dimensions (`d`, `L`, `n_agents`) for faster local verification.
- Save your modified parameters in separate script copies or configuration wrappers to keep experiments traceable.
