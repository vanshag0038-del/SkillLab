"""Evaluate the Champion–Challenger sentiment models and log the comparison."""

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay
from transformers import pipeline, set_seed
from config import ROOT, OUTPUTS, TRACKING_URI, EXPERIMENT, SEED, MODEL_A, MODEL_B, MODEL_REVISIONS, RULE, STUDENT
from hf_model import SentimentModel


def choose_winner(a, b):
    delta = b['macro_f1'] - a['macro_f1']
    return 'B' if delta + 1e-12 >= RULE['margin'] and b['avg_latency_ms'] <= RULE['max_latency_ratio'] * a['avg_latency_ms'] else 'A'


def validate_data(evaluation, new):
    if not {'id', 'text', 'reference'} <= set(evaluation.columns) or not {'id', 'text'} <= set(new.columns):
        raise ValueError('Require evaluation id/text/reference and new-data id/text columns')
    for name, frame in [('Evaluation', evaluation), ('New', new)]:
        if frame['id'].isna().any() or frame['id'].duplicated().any():
            raise ValueError(f'{name} IDs must be present and unique')
        if not frame['text'].map(lambda t: isinstance(t, str) and bool(t.strip())).all():
            raise ValueError(f'{name} texts must be nonblank strings')
        if frame['text'].str.strip().str.casefold().duplicated().any():
            raise ValueError(f'{name} texts must be unique after whitespace/case normalization')
    if evaluation.empty or evaluation['text'].isna().any() or evaluation['text'].duplicated().any():
        raise ValueError('Evaluation data must be nonempty with unique non-null texts')
    if set(evaluation.reference) != {'NEGATIVE', 'POSITIVE'}:
        raise ValueError('Both binary labels are required')
    if not 5 <= len(new) <= 10 or set(new.text.str.strip().str.casefold()) & set(evaluation.text.str.strip().str.casefold()):
        raise ValueError('Require 5–10 new examples disjoint from evaluation')


def main():
    OUTPUTS.mkdir(exist_ok=True)
    set_seed(SEED)
    torch.set_num_threads(4)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    data = pd.read_csv(ROOT / 'data/eval_inputs.csv')
    validate_data(data, pd.read_csv(ROOT / 'data/new_data.csv'))
    versions = {p: importlib.metadata.version(p) for p in ['transformers', 'torch', 'mlflow', 'pandas', 'scikit-learn']}
    readme = (ROOT / 'README.md').read_bytes()
    manifest = {'rule': RULE, 'readme_sha256_before_run': hashlib.sha256(readme).hexdigest(),
                'eval_sha256': hashlib.sha256((ROOT / 'data/eval_inputs.csv').read_bytes()).hexdigest(),
                'library_versions': versions, 'tracking_uri': TRACKING_URI, 'models': {}}
    with mlflow.start_run(run_name='compare_sentiment') as parent:
        manifest['summary_run_id'] = parent.info.run_id
        mlflow.log_params(RULE)
        mlflow.log_text(readme.decode(), 'README_before_run.md')
        for role, model_id, short in [('A', MODEL_A, 'distilbert-sst2'), ('B', MODEL_B, 'bert-tiny-sst2')]:
            print(f'Loading {role}: {model_id}', flush=True)
            pipe = pipeline('text-classification', model=model_id, revision=MODEL_REVISIONS[model_id], device=-1)
            pipe(['Warm-up input.'], truncation=True, max_length=128)
            started = time.perf_counter()
            raw = pipe(data.text.tolist(), batch_size=8, truncation=True, max_length=128)
            elapsed = time.perf_counter() - started
            mapping = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'POSITIVE', 'NEGATIVE': 'NEGATIVE', 'POSITIVE': 'POSITIVE'}
            predictions = data.copy()
            predictions['prediction'] = [mapping[x['label'].upper()] for x in raw]
            predictions['confidence'] = [x['score'] for x in raw]
            predictions['correct'] = predictions.reference == predictions.prediction
            p, r, f, _ = precision_recall_fscore_support(data.reference, predictions.prediction, average='macro', zero_division=0)
            metrics = {'accuracy': accuracy_score(data.reference, predictions.prediction),
                       'macro_precision': p, 'macro_recall': r, 'macro_f1': f,
                       'avg_latency_ms': 1000 * elapsed / len(data), 'total_inference_s': elapsed,
                       'parameters_millions': sum(x.numel() for x in pipe.model.parameters()) / 1e6}
            folder = OUTPUTS / role
            folder.mkdir(exist_ok=True)
            predictions.to_csv(folder / f'predictions_{role}.csv', index=False)
            (folder / 'metrics_summary.json').write_text(json.dumps(metrics, indent=2))
            ConfusionMatrixDisplay.from_predictions(data.reference, predictions.prediction, labels=['NEGATIVE', 'POSITIVE'], cmap='Blues')
            plt.title(f'{role}: {short}')
            plt.tight_layout(); plt.savefig(folder / 'confusion_matrix.png'); plt.close()
            local = ROOT / '.cache' / role
            pipe.save_pretrained(str(local))
            revision = getattr(pipe.model.config, '_commit_hash', None)
            with mlflow.start_run(run_name=f'{role}_sentiment_{short}', nested=True) as run:
                mlflow.log_params({'hf_model_id': model_id, 'hf_revision': revision or 'unknown', 'task': 'sentiment',
                                  'role': 'champion candidate' if role == 'A' else 'challenger candidate',
                                  'max_length': 128, 'batch_size': 8, 'eval_samples': len(data), 'seed': SEED, 'device': 'cpu', 'torch_threads': 4})
                mlflow.set_tags({'student_usn': os.getenv('STUDENT_USN', STUDENT['usn']), 'student_name': STUDENT['name'], 'student_section': STUDENT['section'], 'library_versions': json.dumps(versions), 'dataset_name': 'original-balanced-sentiment-60'})
                mlflow.log_metrics(metrics)
                info = mlflow.pyfunc.log_model(name='model', python_model=SentimentModel(), artifacts={'hf': str(local)},
                    code_paths=[str(ROOT / 'hf_model.py')], input_example=data[['text']].head(2),
                    pip_requirements=[f'{k}=={v}' for k, v in versions.items() if k in ['mlflow', 'transformers', 'torch', 'pandas']],
                    metadata={'hf_model_id': model_id, 'hf_revision': revision})
                mlflow.log_artifacts(str(folder))
                mlflow.log_artifact(str(ROOT / 'data/eval_inputs.csv'))
                mlflow.log_artifact(str(ROOT / 'requirements.txt'))
                mlflow.log_text(readme.decode(), 'README.md')
                manifest['models'][role] = {'run_id': run.info.run_id, 'model_uri': info.model_uri, 'hf_model_id': model_id, 'hf_revision': revision, 'metrics': metrics}
            del pipe
        a, b = [manifest['models'][r]['metrics'] for r in ['A', 'B']]
        winner = choose_winner(a, b)
        manifest['winner'] = winner
        manifest['delta_primary_metric'] = b['macro_f1'] - a['macro_f1']
        mlflow.set_tag('winner', winner)
        mlflow.log_metric('delta_primary_metric', manifest['delta_primary_metric'])
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for ax, key in zip(axes.flat, a):
            ax.bar(['A', 'B'], [a[key], b[key]], color=['#176c91', '#d1772b'])
            ax.set_title(key)
        axes.flat[-1].axis('off'); fig.tight_layout(); fig.savefig(OUTPUTS / 'comparison.png'); plt.close(fig)
        pa, pb = [pd.read_csv(OUTPUTS / r / f'predictions_{r}.csv') for r in ['A', 'B']]
        cases = pa[['id', 'text', 'reference']].assign(prediction_A=pa.prediction, prediction_B=pb.prediction)
        cases = cases[cases.prediction_A != cases.prediction_B].copy()
        cases['comment'] = cases.apply(lambda x: f"{'A' if x.prediction_A == x.reference else 'B'} matches the intended label; examine negation, mixed sentiment and wording.", axis=1)
        cases.to_csv(OUTPUTS / 'disagreements.csv', index=False)
        manifest['disagreements_count'] = len(cases)
        (OUTPUTS / 'metrics_summary.json').write_text(json.dumps(manifest, indent=2))
        rows = '\n'.join(f'|{key}|{a[key]:.6f}|{b[key]:.6f}|{b[key]-a[key]:+.6f}|' for key in a)
        decision = (f"Model {winner} is the Champion under the rule declared before evaluation. "
            f"The Challenger's macro-F1 is {b['macro_f1']:.4f}, compared with {a['macro_f1']:.4f} for A: "
            f"a difference of {100*(b['macro_f1']-a['macro_f1']):+.2f} percentage points. "
            f"The required improvement is one percentage point and the measured latency ratio B/A is {b['avg_latency_ms']/a['avg_latency_ms']:.3f}. "
            "Promotion requires both conditions; a speed advantage alone does not justify replacement under this policy. "
            "The comparison uses the same sixty sentences in the same order, with thirty intended labels in each class. "
            "With only sixty observations, a single changed prediction can move accuracy by about 1.67 percentage points. "
            "Consequently the observed difference should not be interpreted as proof of a population-level improvement or statistical significance. "
            "Macro-F1 gives both classes equal weight, while the confusion matrices show which class accounts for errors. "
            "The tiny model uses fewer parameters and may be useful where memory and CPU throughput matter more than this accuracy policy. "
            "The disagreement table records all observed conflicting predictions for qualitative review. "
            "Before deployment, the decision should be checked on a larger independently labeled sample drawn from actual users, including ambiguous language and domain shifts.")
        result = '|Metric|A|B|B minus A|\n|---|---:|---:|---:|\n' + rows + '\n\n' + decision
        (OUTPUTS / 'results.md').write_text(result, encoding='utf-8')
        text = readme.decode()
        pre, tail = text.split('<!-- RESULTS_START -->')
        _, post = tail.split('<!-- RESULTS_END -->')
        (ROOT / 'README.md').write_text(pre + '<!-- RESULTS_START -->\n' + result + '\n<!-- RESULTS_END -->' + post, encoding='utf-8')
        # The comparison is created after both child runs have closed. Attach it
        # explicitly to each run so a fresh execution has complete evidence.
        client = mlflow.MlflowClient()
        for entry in manifest['models'].values():
            client.log_artifact(entry['run_id'], str(OUTPUTS / 'comparison.png'))
        # Log only evidence created by this comparison, never stale lab stores,
        # document previews or a previous submission archive under outputs/.
        for filename in ['comparison.png', 'disagreements.csv', 'metrics_summary.json', 'results.md']:
            mlflow.log_artifact(str(OUTPUTS / filename), artifact_path='comparison')
        for role in ['A', 'B']:
            mlflow.log_artifacts(str(OUTPUTS / role), artifact_path=f'comparison/{role}')
        mlflow.log_artifact(str(ROOT / 'README.md'))
        print(result, flush=True)


if __name__ == '__main__':
    main()

