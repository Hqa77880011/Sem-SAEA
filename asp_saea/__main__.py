import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.special import ndtr
from scipy.stats import qmc
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize as evolutionary_minimize
from pymoo.util.ref_dirs import get_reference_directions
from sentence_transformers import SentenceTransformer


def nondominated(y):
    return np.array([not np.any(np.all(y <= row, axis=1) & np.any(y < row, axis=1)) for row in y])


class GP:
    def __init__(self, x, y, center, penalty):
        self.x = x
        self.mean, self.scale = y.mean(), max(y.std(), 1e-8)
        self.y = (y - self.mean) / self.scale
        self.center = center
        self.penalty = penalty
        initial = np.r_[np.full(x.shape[1], np.log(.15)), 0., -8.]
        result = minimize(self.objective, initial, method='L-BFGS-B', bounds=[(-5, 3)] * x.shape[1] + [(-5, 5), (-14, -1)])
        self.theta = result.x
        self.factor, self.alpha, _ = self.components(self.theta)

    def kernel(self, a, b, theta):
        distance = cdist(a / np.exp(theta[:-2]), b / np.exp(theta[:-2]))
        scaled = np.sqrt(5) * distance
        return np.exp(theta[-2]) * (1 + scaled + scaled ** 2 / 3) * np.exp(-scaled)

    def components(self, theta):
        matrix = self.kernel(self.x, self.x, theta) + np.exp(theta[-1]) * np.eye(len(self.x))
        factor = cho_factor(matrix, lower=True)
        alpha = cho_solve(factor, self.y)
        nll = .5 * self.y @ alpha + np.log(np.diag(factor[0])).sum() + .5 * len(self.y) * np.log(2 * np.pi)
        return factor, alpha, nll

    def objective(self, theta):
        _, _, nll = self.components(theta)
        return nll + .5 * self.penalty * np.sum((theta[:-2] - self.center) ** 2)

    def predict(self, x):
        k = self.kernel(x, self.x, self.theta)
        mean = k @ self.alpha
        variance = np.maximum(np.exp(self.theta[-2]) - np.sum(k * cho_solve(self.factor, k.T).T, axis=1), 1e-12)
        return self.mean + self.scale * mean, self.scale * np.sqrt(variance)

    def loo(self):
        diagonal = np.diag(cho_solve(self.factor, np.eye(len(self.x))))
        variance = 1 / diagonal
        residual = self.alpha / diagonal
        nlpd = .5 * np.mean(np.log(2 * np.pi * variance) + residual ** 2 / variance)
        prediction = (self.y - residual) * self.scale + self.mean
        return nlpd, prediction


def problem(c):
    if c['problem'] != 'external':
        kwargs = dict(n_var=c['dimension'])
        if c['problem'].startswith(('dtlz', 'wfg')):
            kwargs['n_obj'] = c.get('objectives', 3)
        if c['problem'].startswith('wfg'):
            kwargs['k'] = c.get('position_parameters', 4)
        return get_problem(c['problem'], **kwargs)
    class External(Problem):
        def __init__(self):
            super().__init__(n_var=c['dimension'], n_obj=c['objectives'], n_ieq_constr=c['constraints'],
                             xl=np.array(c['lower']), xu=np.array(c['upper']))

        def _evaluate(self, x, out, *args, **kwargs):
            results = []
            for design in x:
                actual = design.copy()
                actual[c['integer_indices']] = np.rint(actual[c['integer_indices']])
                completed = subprocess.run(c['external_command'], input=json.dumps(actual.tolist()),
                                           text=True, capture_output=True, check=True)
                values = json.loads(completed.stdout)
                results.append(values)
            out['F'] = np.array([r['objectives'] for r in results])
            out['G'] = np.array([r['constraints'] for r in results])
    return External()


def exact(p, x):
    result = p.evaluate(np.atleast_2d(x), return_values_of=['F', 'G'], return_as_dictionary=True)
    f, g = result['F'], result['G']
    if not np.isfinite(f).all() or not np.isfinite(g).all():
        raise ValueError('Exact evaluator returned nonfinite responses')
    return np.concatenate([f, g], axis=1)


def prepare(c, out):
    p = problem(c)
    rng = np.random.default_rng(c['seed'])
    designs = [qmc.LatinHypercube(p.n_var, seed=int(rng.integers(2 ** 31))).random(5 * p.n_var) for _ in range(20)]
    x = max(designs, key=lambda design: pdist(design).min())
    if len(x) >= c['budget']:
        raise ValueError('Budget must exceed 5d initial exact evaluations')
    encoder = SentenceTransformer(c['encoder'])
    variables = c['variable_records']
    responses = c['response_records']
    if len(variables) != p.n_var or len(responses) != p.n_obj + p.n_ieq_constr:
        raise ValueError('One metadata record per variable and objective/constraint is required')
    ex = encoder.encode(variables, normalize_embeddings=True)
    ey = encoder.encode(responses, normalize_embeddings=True)
    relations = (1 + ey @ ex.T) / 2
    start = time.perf_counter()
    y = exact(p, p.xl + x * (p.xu - p.xl))
    np.savez_compressed(out / 'initial.npz', x=x, y=y, relation=relations, exact_seconds=time.perf_counter() - start)


def run(c, out):
    p = problem(c)
    initial = np.load(out / 'initial.npz')
    x, y = initial['x'].copy(), initial['y'].copy()
    relation = initial['relation'].copy()
    rng = np.random.default_rng(c['seed'])
    if c['variant'] == 'shuffled':
        relation = relation[:, rng.permutation(p.n_var)]
    references = get_reference_directions('das-dennis', p.n_obj, n_partitions=c['reference_partitions'])
    rows = []
    while len(x) < c['budget']:
        start = time.perf_counter()
        models, diagnostics = [], []
        for response in range(y.shape[1]):
            center = np.log(.15) + np.log(4) * (1 - relation[response])
            penalty = 2 * np.exp(-len(x) / (5 * p.n_var))
            numerical = GP(x, y[:, response], center, 0)
            semantic = GP(x, y[:, response], center, penalty)
            semantic_nlpd, semantic_loo = semantic.loo()
            numerical_nlpd, numerical_loo = numerical.loo()
            gate = semantic_nlpd <= numerical_nlpd + .05 * max(abs(numerical_nlpd), 1e-8)
            if c['variant'] == 'blank':
                gate = False
            elif c['variant'] == 'no-gate':
                gate = True
            chosen = semantic if gate else numerical
            loo = semantic_loo if gate else numerical_loo
            mse = ((y[:, response] - loo) ** 2).mean()
            variance = y[:, response].var()
            diagnostics.append(dict(response=response, gate=bool(gate), loo_r2=1 - mse / max(variance, 1e-12),
                                    loo_nrmse=np.sqrt(mse / max(variance, 1e-12))))
            models.append(chosen)
        class Surrogate(Problem):
            def __init__(self):
                super().__init__(n_var=p.n_var, n_obj=p.n_obj, xl=0, xu=1)

            def _evaluate(self, designs, out, *args, **kwargs):
                out['F'] = np.column_stack([model.predict(designs)[0] for model in models[:p.n_obj]])
        result = evolutionary_minimize(Surrogate(), RVEA(ref_dirs=references), ('n_gen', c['virtual_generations']),
                                       seed=c['seed'] + len(x), verbose=False)
        pool = np.vstack([result.pop.get('X'), rng.random((c['random_candidates'], p.n_var))])
        mu, sigma = zip(*[model.predict(pool) for model in models])
        mu, sigma = np.array(mu).T, np.array(sigma).T
        feasibility = ndtr(-mu[:, p.n_obj:] / sigma[:, p.n_obj:].clip(1e-9)).prod(1)
        actual_feasible = (y[:, p.n_obj:] <= 0).all(1)
        front = y[actual_feasible, :p.n_obj]
        if len(front):
            front = front[nondominated(front)]
            ideal = np.asarray(c['ideal'])
            nadir = np.asarray(c['nadir'])
            normalized = (front - ideal) / (nadir - ideal)
            hv = HV(ref_point=np.array(c['reference']))
            base = hv(normalized)
            gain = np.array([max(0., hv(np.vstack([normalized, (row[:p.n_obj] - ideal) / (nadir - ideal)])) - base) for row in mu])
        else:
            violation = np.maximum(0, mu[:, p.n_obj:] - 2 * sigma[:, p.n_obj:]).sum(1)
            gain = -violation.argsort().argsort().astype(float)
        uncertainty = (sigma[:, :p.n_obj] / y[:, :p.n_obj].std(0).clip(1e-8)).mean(1)
        distance = cdist(pool, x).min(1)
        normalized_scores = [(s - s.min()) / max(np.ptp(s), 1e-12) for s in [gain, uncertainty, distance]]
        eta = .25 + .65 * len(x) / c['budget']
        acquisition = feasibility * (eta * normalized_scores[0] + (1 - eta) * normalized_scores[1] + .1 * normalized_scores[2])
        acquisition[distance < 1e-8] = -np.inf
        next_x = pool[acquisition.argmax()]
        overhead = time.perf_counter() - start
        start = time.perf_counter()
        observed = exact(p, p.xl + next_x * (p.xu - p.xl))[0]
        exact_seconds = time.perf_counter() - start
        x = np.vstack([x, next_x])
        y = np.vstack([y, observed])
        rows.append(dict(evaluation=len(x), overhead_seconds=overhead, exact_seconds=exact_seconds,
                         diagnostics=json.dumps(diagnostics), prediction=json.dumps(mu[acquisition.argmax()].tolist()),
                         design=json.dumps(next_x.tolist()), response=json.dumps(observed.tolist())))
        np.savez_compressed(out / 'archive.npz', x=x, y=y)
        pd.DataFrame(rows).to_csv(out / 'search.csv', index=False)


def evaluate(c, out):
    p = problem(c)
    archive = np.load(out / 'archive.npz')
    y = archive['y']
    feasible = (y[:, p.n_obj:] <= 0).all(1)
    front = y[feasible, :p.n_obj]
    front = front[nondominated(front)] if len(front) else front
    normalized = (front - c['ideal']) / (np.array(c['nadir']) - c['ideal'])
    result = dict(method=c['variant'], hypervolume=float(HV(ref_point=np.array(c['reference']))(normalized)) if len(front) else 0.,
                  feasible_rate=float(feasible.mean()), exact_evaluations=len(y), optimizer_overhead_seconds=float(pd.read_csv(out / 'search.csv').overhead_seconds.sum()))
    reference_front = np.load(c['reference_front']) if c['reference_front'] else p.pareto_front()
    if reference_front is not None and len(front):
        reference_front = (reference_front - c['ideal']) / (np.array(c['nadir']) - c['ideal'])
        result['igd_plus'] = float(IGDPlus(reference_front)(normalized))
    pd.DataFrame([result]).to_csv(out / 'metrics.csv', index=False)
    np.save(out / 'pareto.npy', front)


def plot(out):
    front = np.load(out / 'pareto.npy')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(front[:, 0], front[:, 1], s=18)
    ax.set(xlabel='Exact objective 1', ylabel='Exact objective 2')
    fig.tight_layout()
    fig.savefig(out / 'pareto.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'run', 'evaluate', 'plot'])
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    p.add_argument('--variant')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.variant:
        c['variant'] = args.variant
    if c['variant'] not in ['full', 'blank', 'shuffled', 'no-gate']:
        p.error('Unknown variant')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    if args.variant is None and settings.exists():
        c['variant'] = json.loads(settings.read_text(encoding='utf-8')).get('variant', c.get('variant', 'full'))
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
