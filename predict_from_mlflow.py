"""Inference knows only a registry name/alias, never a candidate or Hub pipeline."""
import argparse
import mlflow
import pandas as pd
from mlflow import MlflowClient
from config import ROOT, OUTPUTS, TRACKING_URI, REGISTERED_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alias', default='champion', choices=['champion', 'challenger'])
    args = parser.parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, args.alias)
    uri = f'models:/{REGISTERED_MODEL}@{args.alias}'
    model = mlflow.pyfunc.load_model(uri)
    new = pd.read_csv(ROOT / 'data/new_data.csv')
    predictions = model.predict(new[['text']])
    predictions.insert(0, 'id', new.id)
    predictions['model_version'] = version.version
    predictions['hf_model_id'] = version.tags['hf_model_id']
    text = f'Loaded {uri}\nVersion: {version.version}\nHugging Face model: {version.tags["hf_model_id"]}\n\n' + predictions.to_string(index=False)
    print(text)
    OUTPUTS.mkdir(exist_ok=True)
    suffix = '' if args.alias == 'champion' else '_challenger'
    predictions.to_csv(OUTPUTS / f'new_data_predictions{suffix}.csv', index=False)
    (OUTPUTS / f'prediction_terminal{suffix}.txt').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
