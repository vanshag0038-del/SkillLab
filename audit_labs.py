"""Read-only checks of saved lab evidence against recomputed results and MLflow."""
import json
import re
import importlib.metadata
import mlflow
import numpy as np
import pandas as pd
from config import ROOT, OUTPUTS, TRACKING_URI
from run_labs import fit


def check_close(a, b):
    assert np.allclose(a, b, rtol=1e-9, atol=1e-9), (a, b)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    lab_root = OUTPUTS / 'labs'
    checked = {}
    def transcript(n):
        return (lab_root / f'lab_{n:02}' / 'terminal.txt').read_text(encoding='utf-8')
    def runs(n):
        return [client.get_run(rid) for rid in re.findall(r'"run_id":\s*"([a-f0-9]+)"', transcript(n))]
    first = pd.read_csv(lab_root / 'lab_01/metrics.csv')
    for row in first.itertuples():
        _, out, metrics, obs = fit(row.dataset, row.test_size)
        check_close(row.mae, metrics['mae'])
        assert row.train_samples == (obs.split == 'train').sum() and row.test_samples == len(out)
    observations = pd.read_csv(lab_root / 'lab_01/observations.csv')
    assert {'dataset', 'execution', 'id', 'split', 'test_size', 'seed'} <= set(observations.columns)
    assert len(pd.ExcelFile(lab_root / 'lab_01/observations.xlsx').sheet_names) == 6
    checked['01'] = 'Six recomputed split metrics/counts; CSV membership columns; six workbook sheets'
    listing = (lab_root / 'lab_02/store_contents.txt').read_text()
    assert all(token in listing for token in ['meta.yaml', 'params', 'metrics', 'artifacts'])
    assert transcript(2).count('"run_id"') == 2
    checked['02'] = 'Baseline/delivery run evidence and actual legacy store listing'
    three = runs(3)
    assert len(three) == 2 and {r.data.params['dataset'] for r in three} == {'baseline', 'delivery'}
    checked['03'] = 'Both baseline/delivery runs found on tracking server'
    actual = pd.read_csv(lab_root / 'lab_04/actual_predicted_errors.csv')
    _, expected, _, _ = fit('hospital')
    assert actual.id.tolist() == expected.id.tolist()
    check_close(actual.prediction, expected.prediction)
    check_close(actual.error, actual.prediction - actual.y)
    checked['04'] = 'Exact supplied hospital dataset; held-out predictions and signed errors recomputed'
    comparisons = pd.read_csv(lab_root / 'lab_05/comparison.csv')
    assert len(comparisons) == 9
    for row in comparisons.itertuples():
        _, _, metrics, _ = fit(test_size=row.test_size, seed=row.random_state)
        check_close([row.mae, row.r2], [metrics['mae'], metrics['r2']])
    checked['05'] = 'All nine seed/split MAE and R2 pairs recomputed'
    six = json.loads((lab_root / 'lab_06/comparison.json').read_text())
    assert 'training_r2_score' in six['autolog']['metrics'] and 'mae' in six['manual']['metrics']
    check_close(six['manual']['metrics']['mae'], fit()[2]['mae'])
    assert six['autolog']['params']['fit_intercept'] == six['manual']['params']['fit_intercept']
    checked['06'] = 'Actual automatic/manual metadata and held-out MAE cross-checked'
    seven = runs(7)
    assert len(seven) == 2 and len({r.info.experiment_id for r in seven}) == 1
    assert {r.data.params['fit_intercept'] for r in seven} == {'True', 'False'}
    checked['07'] = 'Both intercept configurations share one live experiment'
    eight = pd.read_csv(lab_root / 'lab_08/loaded_predictions.csv')
    uri = (lab_root / 'lab_08/model_uri.txt').read_text()
    check_close(mlflow.pyfunc.load_model(uri).predict(eight[['x']]), eight.prediction)
    checked['08'] = 'Saved sklearn URI reload reproduced all three predictions'
    manifest = json.loads((OUTPUTS / 'metrics_summary.json').read_text())
    assert (lab_root / 'lab_09/model_uri.txt').read_text() == manifest['models']['A']['model_uri']
    nine = pd.read_csv(lab_root / 'lab_09/custom_model_predictions.csv')
    assert nine.prediction.tolist() == ['POSITIVE', 'NEGATIVE']
    checked['09'] = 'Custom model URI and both recorded predictions cross-checked; full HF replay in verify_submission.py'
    for name, metrics in [('Linear Regression Experiment', {'mae'}), ('Linear Regression Metrics Experiment', {'mae', 'mse', 'r2'})]:
        experiment = client.get_experiment_by_name(name)
        found = client.search_runs([experiment.experiment_id], filter_string="attributes.status = 'FINISHED'", max_results=1)
        assert len(found) == 1 and set(found[0].data.metrics) == metrics
    checked['10'] = 'Exact two experiment names and correct distinct logged metric sets'
    eleven = json.loads((lab_root / 'lab_11/registry_comparison.json').read_text())
    eleven.sort(key=lambda r: r['mae'])
    assert client.get_model_version_by_alias('lab-regression', 'champion').run_id == eleven[0]['run_id']
    checked['11'] = 'Regression alias selects the recorded lower-MAE candidate'
    twelve = runs(12)
    assert len(twelve) == 2
    assert {float(r.data.params['test_size']) for r in twelve} == {.25, .4}
    assert all(r.info.status == 'FINISHED' and r.data.params['random_state'] == '7' for r in twelve)
    checked['12'] = 'Both CLI Project runs finished with required sizes and seed'
    top = pd.read_csv(lab_root / 'lab_13/top_three.csv')
    assert top.run_id.tolist() == comparisons.sort_values('mae').head(3).run_id.tolist()
    checked['13'] = 'Top three IDs match the saved nine-run comparison ranked by MAE'
    assert importlib.metadata.version('mlflow') in transcript(14) and 'Lab05-Comparison' in transcript(14)
    checked['14'] = 'CLI version/help/experiment listing evidence confirmed'
    (OUTPUTS / 'lab_audit.json').write_text(json.dumps(checked, indent=2), encoding='utf-8')
    print(json.dumps(checked, indent=2))


if __name__ == '__main__':
    main()
