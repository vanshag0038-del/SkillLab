"""Portable MLflow wrapper: bundled model/tokenizer, no Hub needed on reload."""
import mlflow.pyfunc
import pandas as pd


class SentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, batch_size=8, max_length=128):
        self.batch_size = batch_size
        self.max_length = max_length

    def load_context(self, context):
        from transformers import pipeline
        self.pipe = pipeline('text-classification', model=context.artifacts['hf'],
                             tokenizer=context.artifacts['hf'], device=-1)

    def predict(self, context, model_input, params=None):
        texts = model_input['text'].astype(str).tolist()
        results = self.pipe(texts, batch_size=self.batch_size,
                            truncation=True, max_length=self.max_length)
        mapping = {'LABEL_0': 'NEGATIVE', 'LABEL_1': 'POSITIVE',
                   'NEGATIVE': 'NEGATIVE', 'POSITIVE': 'POSITIVE'}
        return pd.DataFrame({'text': texts,
                             'prediction': [mapping[r['label'].upper()] for r in results],
                             'confidence': [r['score'] for r in results]})
