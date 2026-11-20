"""Shared configuration. Set STUDENT_USN and MLFLOW_TRACKING_URI as needed."""
import os
import sys
import json
from pathlib import Path

# MLflow emits Unicode run links; Windows redirected consoles may default to cp1252.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

ROOT = Path(__file__).resolve().parent
STUDENT = json.loads((ROOT / 'student.json').read_text(encoding='utf-8'))
OUTPUTS = ROOT / 'outputs'
TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
EXPERIMENT = 'hf-sentiment-champion-challenger'
REGISTERED_MODEL = 'sentiment-model'
SEED = 42
MODEL_A = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
MODEL_B = 'M-FAC/bert-tiny-finetuned-sst2'
MODEL_REVISIONS = {
    MODEL_A: '714eb0fa89d2f80546fda750413ed43d93601a13',
    MODEL_B: '41ad6709ec46b414749b37daf49cf5ca1c7dba7c',
}
RULE = {'primary_metric': 'macro_f1', 'margin': 0.01, 'max_latency_ratio': 5.0}
