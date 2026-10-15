# Sem-SAEA

Implementation of **Adaptive semantic priors for surrogate-assisted evolutionary optimization with expensive objectives and constraints.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

The default is numerical ZDT1 through [pymoo](https://pymoo.org/problems/index.html). Metadata records are neutral variable/response descriptions fixed before outcomes. `prepare` caches MiniLM embeddings and a common maximin-selected Latin hypercube with exact responses. For DTLZ/WFG, update the problem, dimensions, objectives, metadata, normalizers and reference front together. For an engineering evaluator set `problem=external` and supply `dimension,objectives,constraints,lower,upper,integer_indices,external_command`; the executable reads one design JSON array on stdin and returns JSON `objectives`/`constraints` on stdout. Failed evaluations raise an error rather than create response values.

## Workflow

```bash
python -m asp_saea prepare
python -m asp_saea run
python -m asp_saea evaluate
python -m asp_saea plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`budget` is exact evaluations, including the initial `5d` designs. `encoder,variable_records,response_records` define semantic evidence. `virtual_generations,reference_partitions,random_candidates` define virtual candidate search. `ideal,nadir,reference` fix HV normalization; `reference_front` supplies independent IGD+ reference points. `variant` controls semantic use. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`initial.npz` stores the paired initial archive and relations. `archive.npz` stores only exact evaluations. `search.csv` records predictions made before evaluation, gate/LOO diagnostics, exact-response time and optimizer overhead. `pareto.npy` contains exact feasible nondominated responses. `metrics.csv` reports HV, IGD+ when a reference exists, feasible rate, budget and overhead.

## Method and comparisons

Independent Matérn-5/2 ARD GPs optimize marginal likelihood with a response-specific semantic log-length-scale penalty. Archive-size decay and leave-one-out predictive density determine whether each semantic prior remains active. RVEA generates virtual candidates; feasibility-weighted HV gain, uncertainty and distance choose one exact query. Only exact feasible responses enter the returned Pareto archive. External CFD remains an exact evaluator interface, not a numerical stand-in for CFD.

Controls: `full`, `blank`, `shuffled`, `no-gate`. Choose `--variant` and use a separate training/output directory.

```bash
python -m asp_saea.compare --inputs "runs/*/metrics.csv" --reference full --metrics hypervolume feasible_rate optimizer_overhead_seconds --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

