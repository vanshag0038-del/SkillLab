"""Executable record for Labs 01–14. Run from the repository root.

Missing instructor starter files are reconstructed using the supplied tables.
All reported numbers are generated, never pre-filled.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from config import ROOT, TRACKING_URI

LABOUT = ROOT / 'outputs' / 'labs'


def dataset(kind='baseline'):
    if kind == 'delivery':
        return pd.DataFrame({'id': range(1, 6), 'x': [1, 2, 3, 4, 5], 'y': [16, 19, 27, 29, 34]})
    if kind == 'hospital':
        return pd.DataFrame({'id': [f'P{i:03}' for i in range(1, 21)],
            'x': [1,2,3,4,5,6,7,8,9,10,2,4,6,8,3,5,7,9,1,10],
            'y': [2,3,3,5,5,7,7,8,10,9,2,4,6,9,4,6,6,9,3,11]})
    # Deterministic educational data because baseline_model.py was not supplied.
    x = np.arange(1, 51)
    return pd.DataFrame({'id': x, 'x': x, 'y': 10 + 2.5*x + np.random.default_rng(42).normal(0, 4, len(x))})


def fit(kind='baseline', test_size=.2, seed=42, intercept=True):
    data = dataset(kind)
    train, test = train_test_split(data, test_size=test_size, random_state=seed)
    model = LinearRegression(fit_intercept=intercept).fit(train[['x']], train.y)
    out = test.copy()
    out['prediction'] = model.predict(test[['x']])
    out['error'] = out.prediction - out.y
    out['absolute_error'] = out.error.abs()
    metrics = {'mae': mean_absolute_error(out.y, out.prediction), 'mse': mean_squared_error(out.y, out.prediction)}
    if len(test) >= 2:
        metrics['r2'] = r2_score(out.y, out.prediction)
    observations = pd.concat([train.assign(split='train'), test.assign(split='test')])
    return model, out, metrics, observations


def track(experiment, name, kind='baseline', test_size=.2, seed=42, intercept=True, full=True, model_log=True):
    model, out, metrics, obs = fit(kind, test_size, seed, intercept)
    if os.getenv('MLFLOW_RUN_ID'):
        # MLflow Projects creates its run before invoking this entry point.
        mlflow.set_experiment(experiment_id=MlflowClient().get_run(os.environ['MLFLOW_RUN_ID']).info.experiment_id)
    else:
        mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name) as run:
        params = {'test_size': test_size, 'random_state': seed, 'fit_intercept': intercept,
                  'dataset': kind, 'train_samples': int((obs.split == 'train').sum()), 'test_samples': len(out)}
        # Projects already logs raw parameter strings (e.g. 0.40); do not overwrite.
        existing = MlflowClient().get_run(run.info.run_id).data.params
        mlflow.log_params({k: v for k, v in params.items() if k not in existing})
        mlflow.log_metrics(metrics if full else {'mae': metrics['mae']})
        mlflow.log_table(out, 'actual_predicted_errors.json')
        mlflow.log_table(obs, 'observations.json')
        if model_log:
            info = mlflow.sklearn.log_model(model, name='model', input_example=pd.DataFrame({'x': [2., 6.]}))
            uri = info.model_uri
        else:
            uri = None
        result = {'run_id': run.info.run_id, 'model_uri': uri, **metrics,
                  'train_samples': int((obs.split == 'train').sum()), 'test_samples': len(out),
                  'test_size': test_size, 'random_state': seed, 'fit_intercept': intercept}
    print(json.dumps(result, indent=2))
    return result, out, obs


def run_lab(number, intercept=None):
    folder = LABOUT / f'lab_{number:02}'
    folder.mkdir(parents=True, exist_ok=True)
    if number == 1:
        summaries, observations = [], []
        with pd.ExcelWriter(folder / 'observations.xlsx') as writer:
            for kind in ['baseline', 'delivery']:
                for size in [.2, .3, .4]:
                    _, out, metrics, obs = fit(kind, size)
                    key = f'{kind}_{size}'
                    obs = obs.assign(execution=key, dataset=kind, test_size=size, seed=42)
                    obs.to_excel(writer, sheet_name=key, index=False)
                    observations.append(obs)
                    summaries.append({'dataset': kind, 'test_size': size, 'train_samples': int((obs.split=='train').sum()), 'test_samples': len(out), **metrics})
        pd.concat(observations).to_csv(folder / 'observations.csv', index=False)
        summary = pd.DataFrame(summaries); summary.to_csv(folder / 'metrics.csv', index=False)
        print(summary.to_string(index=False))
        print('Use execution, dataset, test_size, seed, row id and split columns to preserve changing splits in one CSV. CSV has no sheets; XLSX supplies separate worksheets. R2 is undefined with one test row.')
    elif number == 2:
        # Genuine legacy directory contents, separate from SQL-backed server labs.
        uri = (folder / 'mlruns').as_uri()
        os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        mlflow.set_tracking_uri(uri)
        for kind in ['baseline', 'delivery']:
            track('Lab02-Introduction', kind, kind, model_log=False)
        listing = '\n'.join(str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file())
        (folder / 'store_contents.txt').write_text(listing)
        print(listing)
        print('Experiment meta.yaml describes the experiment. Each run has meta.yaml plus params, metrics and tags files; artifacts hold tables and files. New MLflow servers instead use SQLite for metadata and a separate artifact store.')
        mlflow.set_tracking_uri(TRACKING_URI)
    elif number == 3:
        for kind in ['baseline', 'delivery']:
            track('Lab03-Tracking', kind, kind)
        print('Tracking URI:', mlflow.get_tracking_uri())
    elif number == 4:
        result, out, _ = track('Lab04-Logging', 'hospital-stay', 'hospital')
        out.to_csv(folder / 'actual_predicted_errors.csv', index=False)
        print(out.to_string(index=False))
        (folder / 'counts.json').write_text(json.dumps({k:v for k,v in result.items() if k.endswith('samples')}, indent=2))
    elif number == 5:
        results = []
        for seed in [7, 21, 42]:
            for size in [.2, .3, .4]:
                r, _, _ = track('Lab05-Comparison', f'seed-{seed}_test-{size}', test_size=size, seed=seed, model_log=False)
                results.append(r)
        frame = pd.DataFrame(results); frame.to_csv(folder / 'comparison.csv', index=False)
        print(frame.to_string(index=False))
        print('Compare test_size within the same seed first. More training rows do not guarantee lower error: the held-out examples also change. R2 and MAE fluctuate with split difficulty; do not claim a causal trend from three splits.')
    elif number == 6:
        mlflow.set_experiment('Lab06-Autolog-vs-Manual')
        mlflow.sklearn.autolog(log_models=True)
        with mlflow.start_run(run_name='autolog-linear-regression') as run:
            model, out, _, _ = fit()
            # Public score call lets autolog capture a test score too.
            model.score(out[['x']], out.y)
            auto_id = run.info.run_id
        mlflow.sklearn.autolog(disable=True)
        manual, _, _ = track('Lab06-Autolog-vs-Manual', 'manual-linear-regression')
        client = MlflowClient()
        comparison = {name: {'params': client.get_run(rid).data.params, 'metrics': client.get_run(rid).data.metrics}
                      for name, rid in [('autolog', auto_id), ('manual', manual['run_id'])]}
        (folder / 'comparison.json').write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2))
    elif number == 7:
        for flag in ([intercept] if intercept is not None else [True, False]):
            track('Lab07-Shared-Server', f'intercept-{flag}', intercept=flag)
        print('Both scripts use the same server and experiment. No separate server is needed. A shared store centralizes comparisons, provenance and artifact access for a team.')
    elif number == 8:
        r, _, _ = track('Lab08-Models', 'model-component')
        loaded = mlflow.pyfunc.load_model(r['model_uri'])
        new = pd.DataFrame({'x': [2., 6., 12.]})
        new['prediction'] = loaded.predict(new)
        new.to_csv(folder / 'loaded_predictions.csv', index=False)
        (folder / 'model_uri.txt').write_text(r['model_uri'])
        print(new.to_string(index=False))
    elif number == 9:
        # Reuses the actual logged custom HF model, with its recorded source URI.
        manifest = json.loads((ROOT / 'outputs/metrics_summary.json').read_text())
        uri = manifest['models']['A']['model_uri']
        print('Custom Hugging Face pyfunc URI:', uri)
        model = mlflow.pyfunc.load_model(uri)
        out = model.predict(pd.DataFrame({'text': ['A wonderful surprise.', 'A miserable experience.']}))
        out.to_csv(folder / 'custom_model_predictions.csv', index=False)
        (folder / 'model_uri.txt').write_text(uri)
        print(out.to_string(index=False))
    elif number == 10:
        track('Linear Regression Experiment', 'mae-only', full=False)
        track('Linear Regression Metrics Experiment', 'mae-mse-r2', full=True)
    elif number == 11:
        client = MlflowClient()
        candidates = []
        for flag in [True, False]:
            r, _, _ = track('Lab11-Registry', f'intercept-{flag}', intercept=flag)
            version = mlflow.register_model(r['model_uri'], 'lab-regression')
            candidates.append({**r, 'version': version.version})
        candidates.sort(key=lambda x: x['mae'])
        for alias, row in zip(['champion', 'challenger'], candidates):
            client.set_registered_model_alias('lab-regression', alias, row['version'])
        v = client.get_model_version_by_alias('lab-regression', 'champion')
        loaded = mlflow.pyfunc.load_model('models:/lab-regression@champion')
        print('Loaded Champion version', v.version, loaded.predict(pd.DataFrame({'x': [3., 7.]})))
        (folder / 'registry_comparison.json').write_text(json.dumps(candidates, indent=2))
    elif number == 12:
        print((ROOT / 'labs/lab_12/MLproject').read_text())
        for size in ['0.25', '0.40']:
            env = {**os.environ, 'MLFLOW_TRACKING_URI': mlflow.get_tracking_uri(), 'MLFLOW_EXPERIMENT_NAME': 'Lab12-Projects', 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']}
            result = subprocess.run([sys.executable, '-m', 'mlflow', 'run', str(ROOT / 'labs/lab_12'), '--env-manager', 'local', '-P', f'test_size={size}', '-P', 'random_state=7'], env=env, capture_output=True, text=True, encoding='utf-8')
            print(result.stdout, result.stderr)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
    elif number == 13:
        client = MlflowClient()
        experiment = client.get_experiment_by_name('Lab05-Comparison')
        if experiment is None:
            raise RuntimeError('Run Lab05 first; use the same tracking URI, never copy a live database.')
        runs = client.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'", order_by=['metrics.mae ASC'], max_results=3)
        rows = [{'run_id': r.info.run_id, 'mae': r.data.metrics['mae'], **r.data.params} for r in runs]
        pd.DataFrame(rows).to_csv(folder / 'top_three.csv', index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    elif number == 14:
        for command in [['--version'], ['--help'], ['experiments', 'search']]:
            result = subprocess.run([sys.executable, '-m', 'mlflow', *command], check=True, capture_output=True, text=True, encoding='utf-8', env={**os.environ, 'MLFLOW_TRACKING_URI': mlflow.get_tracking_uri()})
            print(result.stdout)
    else:
        raise ValueError('Lab number must be 1 through 14')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lab', type=int)
    parser.add_argument('--intercept', choices=['true', 'false'])
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    for lab in ([args.lab] if args.lab else range(1, 15)):
        folder = LABOUT / f'lab_{lab:02}'
        folder.mkdir(parents=True, exist_ok=True)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            run_lab(lab, None if args.intercept is None else args.intercept == 'true')
        output = capture.getvalue()
        (folder / 'terminal.txt').write_text(output, encoding='utf-8')
        print(f'LAB {lab:02}\n{output}', flush=True)


if __name__ == '__main__':
    main()
