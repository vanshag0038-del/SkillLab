"""Register both evaluated models; no hard-coded run IDs."""
import json
import mlflow
from mlflow import MlflowClient
from config import OUTPUTS, TRACKING_URI, REGISTERED_MODEL


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    result = json.loads((OUTPUTS / 'metrics_summary.json').read_text())
    versions = {}
    for role, model in result['models'].items():
        version = mlflow.register_model(model['model_uri'], REGISTERED_MODEL)
        client.set_model_version_tag(REGISTERED_MODEL, version.version, 'hf_model_id', model['hf_model_id'])
        client.set_model_version_tag(REGISTERED_MODEL, version.version, 'candidate', role)
        versions[role] = version.version
    for role, version in versions.items():
        alias = 'champion' if role == result['winner'] else 'challenger'
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version)
        print(f'{REGISTERED_MODEL}@{alias} -> version {version} ({role})')
    (OUTPUTS / 'registry.json').write_text(json.dumps({'name': REGISTERED_MODEL, 'versions': versions, 'winner': result['winner']}, indent=2))


if __name__ == '__main__':
    main()
