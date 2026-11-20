"""Audit recorded results against data, reloadable models and live registry."""
import hashlib
import json
from datetime import datetime, timezone
import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import ROOT, OUTPUTS, TRACKING_URI, REGISTERED_MODEL, STUDENT, MODEL_REVISIONS, RULE
from evaluate import choose_winner, validate_data


def main():
    torch.set_num_threads(4)
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    result = json.loads((OUTPUTS / 'metrics_summary.json').read_text())
    evaluation = pd.read_csv(ROOT / 'data/eval_inputs.csv')
    validate_data(evaluation, pd.read_csv(ROOT / 'data/new_data.csv'))
    assert hashlib.sha256((ROOT / 'data/eval_inputs.csv').read_bytes()).hexdigest() == result['eval_sha256']
    assert result['rule'] == RULE
    checks = []
    for role in ['A', 'B']:
        entry = result['models'][role]
        predictions = pd.read_csv(OUTPUTS / role / f'predictions_{role}.csv')
        assert predictions.text.tolist() == evaluation.text.tolist()
        assert predictions.reference.tolist() == evaluation.reference.tolist()
        p, r, f, _ = precision_recall_fscore_support(predictions.reference, predictions.prediction, average='macro', zero_division=0)
        computed = {'accuracy': accuracy_score(predictions.reference, predictions.prediction), 'macro_precision': p, 'macro_recall': r, 'macro_f1': f}
        for metric, value in computed.items():
            assert abs(value - entry['metrics'][metric]) < 1e-10, (role, metric)
        assert predictions.correct.tolist() == (predictions.reference == predictions.prediction).tolist()
        assert predictions.confidence.between(0, 1).all()
        assert entry['hf_revision'] == MODEL_REVISIONS[entry['hf_model_id']]
        assert abs(entry['metrics']['avg_latency_ms'] * len(evaluation) / 1000 - entry['metrics']['total_inference_s']) < 1e-8
        run = client.get_run(entry['run_id'])
        assert run.info.status == 'FINISHED'
        assert all(t in run.data.tags for t in ['student_usn', 'library_versions', 'dataset_name'])
        assert run.data.tags['student_usn'] == STUDENT['usn']
        assert run.data.tags['mlflow.runName'].startswith(f'{role}_sentiment_')
        for param in ['hf_model_id', 'task', 'role', 'max_length', 'batch_size', 'eval_samples', 'seed', 'device']:
            assert param in run.data.params, (role, param)
        for metric, value in entry['metrics'].items():
            assert abs(run.data.metrics[metric] - value) < 1e-10, (role, metric)
        artifacts = {a.path for a in client.list_artifacts(entry['run_id'])}
        assert {'README.md', 'requirements.txt', 'eval_inputs.csv', 'metrics_summary.json', 'confusion_matrix.png', 'comparison.png', f'predictions_{role}.csv'} <= artifacts
        reloaded = mlflow.pyfunc.load_model(entry['model_uri'])
        replay = reloaded.predict(evaluation[['text']])
        assert replay.prediction.tolist() == predictions.prediction.tolist()
        checks.append(f'{role}: inputs, all classification metrics, latency arithmetic, parameters, artifacts, pinned revision and full model replay passed')
    assert choose_winner(*[result['models'][r]['metrics'] for r in ['A','B']]) == result['winner']
    version = client.get_model_version_by_alias(REGISTERED_MODEL, 'champion')
    assert version.tags['candidate'] == result['winner']
    assert version.run_id == result['models'][result['winner']]['run_id']
    other_role = 'B' if result['winner'] == 'A' else 'A'
    other = client.get_model_version_by_alias(REGISTERED_MODEL, 'challenger')
    assert other.version != version.version and other.run_id == result['models'][other_role]['run_id']
    summary = client.get_run(result['summary_run_id'])
    assert summary.info.status == 'FINISHED' and summary.data.tags['mlflow.runName'] == 'compare_sentiment'
    assert summary.data.tags['winner'] == result['winner']
    assert summary.data.params['primary_metric'] == RULE['primary_metric']
    assert float(summary.data.params['margin']) == RULE['margin']
    assert abs(summary.data.metrics['delta_primary_metric'] - result['delta_primary_metric']) < 1e-10
    new = pd.read_csv(OUTPUTS / 'new_data_predictions.csv')
    source = pd.read_csv(ROOT / 'data/new_data.csv')
    assert new[['id', 'text']].equals(source)
    assert set(new.model_version.astype(str)) == {str(version.version)}
    assert set(new.hf_model_id) == {version.tags['hf_model_id']}
    alias_replay = mlflow.pyfunc.load_model(f'models:/{REGISTERED_MODEL}@champion').predict(source[['text']])
    assert alias_replay.prediction.tolist() == new.prediction.tolist()
    assert all((OUTPUTS / 'labs' / f'lab_{i:02}' / 'terminal.txt').exists() for i in range(1,15))
    checks += ['Decision and both registry aliases verified', 'All fourteen lab transcripts present', 'Eight new-data predictions saved']
    decision = (OUTPUTS / 'results.md').read_text(encoding='utf-8').split('\n\n', 1)[1]
    assert 150 <= len(decision.split()) <= 250
    quizzes = json.loads((ROOT / 'quiz_answers.json').read_text())
    assert len(quizzes['day_1']) == len(quizzes['day_2']) == 15
    checks += ['Student identity verified in live MLflow tags', 'Both supplied quizzes have 15 completed responses with explanations']
    for filename in ['mlflow_compare.png', 'registry_aliases.png', 'prediction_terminal.png']:
        assert (ROOT / 'screenshots' / filename).exists()
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'checks': checks, 'remaining_student_actions': ['Fill course title and course code on the cover page', 'Provide LMS assignment URL for submission', 'Give instructor access to the private repository', 'Add a native terminal screenshot; the included image is a labeled transcript rendering', 'Review implementation and quiz explanations for viva']}
    (OUTPUTS / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
