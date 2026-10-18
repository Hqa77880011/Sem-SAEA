# Sem-SAEA

Implementation of **Adaptive semantic priors for surrogate-assisted evolutionary optimization with expensive objectives and constraints.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run. pymoo/sentence-transformers runtime dependencies were not installed in this inspection environment.

## Data and preprocessing

The default is numerical ZDT1 through [pymoo](https://pymoo.org/problems/index.html). Metadata records are neutral variable/response descriptions fixed before outcomes. `prepare` caches MiniLM embeddings and a common maximin-selected Latin hypercube with exact responses. For DTLZ/WFG, update the problem, dimensions, objectives, metadata, normalizers and reference front together. For an engineering evaluator set `problem=external` and supply `dimension,objectives,constraints,lower,upper,integer_indices,external_command`; the executable reads one design JSON array on stdin and returns JSON `objectives`/`constraints` on stdout. Failed evaluations raise an error rather than create response values.

```bash
python -m asp_saea prepare --config config.json --output runs/full
```

## Run

```bash
python -m asp_saea run --config config.json --output runs/full
```

`config.json` records the method settings and seed `2026001`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m asp_saea evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `blank`, `shuffled`, `no-gate`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m asp_saea prepare --config config.json --output runs/blank --variant blank
python -m asp_saea run --config config.json --output runs/blank --variant blank
python -m asp_saea evaluate --config config.json --output runs/blank --variant blank
```

## Outputs

All main outputs are written below `runs/full/`.

`initial.npz` stores the paired initial archive and relations. `archive.npz` stores only exact evaluations. `search.csv` records predictions made before evaluation, gate/LOO diagnostics, exact-response time and optimizer overhead. `pareto.npy` contains exact feasible nondominated responses. `metrics.csv` reports HV, IGD+ when a reference exists, feasible rate, budget and overhead.

```bash
python -m asp_saea plot --config config.json --output runs/full
```
