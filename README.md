# MLflow Skill Lab: all 14 labs and Champion–Challenger project

Student name: HARSHA BS  
USN: 1MS25CI051  
Section: A  
Course title/code: blank on the record cover; fill in the official course details.

## Task and models

Binary English sentiment classification on 60 fixed, balanced, original labeled
sentences. A (current Champion): `distilbert/distilbert-base-uncased-finetuned-sst-2-english`.
B (Challenger): `M-FAC/bert-tiny-finetuned-sst2`. Both are pretrained Hugging Face
sentiment models. The brief permits substitute models for the same task; the tiny
Challenger explores quality versus CPU efficiency. Data provenance: `data/README.md`.

## Champion rule — declared before execution

Primary metric: macro-F1, higher is better. Promote B only if B minus A macro-F1
is at least **0.01** AND B's warm mean per-example latency is at most **5 times** A's.
Otherwise retain A, including exact ties and differences below the margin.
The one percentage-point margin avoids replacing a model for a negligible score
change; it is an operational threshold, not a statistical significance test. The
latency limit prevents an excessive runtime regression. Accuracy, macro precision,
macro recall and F1 are higher-is-better; latency and model size are lower-is-better.
This text and `config.RULE` are fixed before the final run. A SHA-256 copy of this
README and the input data is logged with every comparison.

## How to run (PowerShell, from repository root)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c requirements-lock.txt
$env:STUDENT_USN = '1MS25CI051'
.\.venv\Scripts\python.exe -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Keep that terminal running. In a second terminal:

```powershell
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe register_champion.py
.\.venv\Scripts\python.exe predict_from_mlflow.py
.\.venv\Scripts\python.exe run_labs.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe verify_submission.py
.\.venv\Scripts\python.exe audit_labs.py
```

First downloads need Internet; saved MLflow models bundle tokenizer and weights
for offline reload. CPU, seed 42, batch size 8 and maximum 128 tokens are fixed.
Model download/load and one warm-up call are excluded from timing; elapsed time
is measured around evaluation inference only. Full environment is saved to
`requirements-lock.txt`. See `labs/README.md` for each exercise and commands.
The setup above creates the environment on a fresh clone. It was tested on
Windows with CPython 3.13. Both Hub model revisions are pinned in config.py to
the exact commits recorded in outputs/metrics_summary.json. The lock file is
the executed environment snapshot; verification does not rewrite it.

The submitted ZIP contains saved results, source and documents. Tracking databases
and model weights are intentionally excluded, as required by the brief. On another
machine, run the commands above in order to recreate the tracking store and model
versions before live verification. Historical run IDs in the saved evidence refer
to the original execution. First-time Hub model downloads require network access.
Optional `--alias challenger` inference saves separate files with `_challenger`
in their names and preserves the Champion's evidence.

## Results table and decision

Written from measured outputs by `evaluate.py` into `outputs/results.md`, and
included below after execution. The decision follows the rule above unchanged.

<!-- RESULTS_START -->
|Metric|A|B|B minus A|
|---|---:|---:|---:|
|accuracy|0.966667|0.800000|-0.166667|
|macro_precision|0.968750|0.837500|-0.131250|
|macro_recall|0.966667|0.800000|-0.166667|
|macro_f1|0.966630|0.794286|-0.172344|
|avg_latency_ms|8.623835|0.870680|-7.753155|
|total_inference_s|0.517430|0.052241|-0.465189|
|parameters_millions|66.955010|4.386178|-62.568832|

Model A is the Champion under the rule declared before evaluation. The Challenger's macro-F1 is 0.7943, compared with 0.9666 for A: a difference of -17.23 percentage points. The required improvement is one percentage point and the measured latency ratio B/A is 0.101. Promotion requires both conditions; a speed advantage alone does not justify replacement under this policy. The comparison uses the same sixty sentences in the same order, with thirty intended labels in each class. With only sixty observations, a single changed prediction can move accuracy by about 1.67 percentage points. Consequently the observed difference should not be interpreted as proof of a population-level improvement or statistical significance. Macro-F1 gives both classes equal weight, while the confusion matrices show which class accounts for errors. The tiny model uses fewer parameters and may be useful where memory and CPU throughput matter more than this accuracy policy. The disagreement table records all observed conflicting predictions for qualitative review. Before deployment, the decision should be checked on a larger independently labeled sample drawn from actual users, including ambiguous language and domain shifts.
<!-- RESULTS_END -->

## Limitations and declaration

This convenience sample is small, balanced and not randomly drawn from a target
population. Scores may reflect dataset construction and pretrained training overlap
cannot be ruled out. CPU timing varies with machine load and batching; it is not a
production benchmark. Confidence scores are not calibrated probabilities. Both
models may miss sarcasm, context or domain-specific language. No test data is used
for training or threshold selection. AI assistance (OpenAI Codex) was used to read
requirements, author data, implement code, debug and write explanations. The student
must review and understand every line and explain the choices in
the viva. The two subsequently supplied quiz documents are answered in
`Completed_Quiz_Answers.docx` and `quiz_answers.json`, with explanations checked
against their supplied keys. No instructor marks are claimed.

## Submission

Repository: [1MS25CI051-mlflow-champion-challenger](https://github.com/harsha31bs07-blip/1MS25CI051-mlflow-champion-challenger) (private).
This final revision is prepared locally; it has not been pushed or submitted to
the LMS. Instructor access and LMS submission remain to be completed. Current
state is recorded in `submission_status.json` and `SUBMISSION_GUIDE.md`.

Use repository name `<USN>-mlflow-champion-challenger`. Commit source/data/outputs
but exclude `.venv`, model weights, tracking databases, `mlruns` and `mlartifacts`.
Capture the three genuine screenshots described in `screenshots/README.md`.
GitHub publication and LMS submission require the student's account and identity.

## Completed local evidence

`Completed_Lab_Record.docx` (and PDF) contains the lab observations and actual
output excerpts. Full transcripts and tables are under `outputs/labs/`.
`outputs/verification.json` records full evaluation replay for both MLflow models,
metric recalculation and registry checks. Run pytest for the current test count.
The three screenshot files are under `screenshots/`; the prediction image is a
clearly labeled rendering of the verbatim terminal transcript. Capture the native
terminal window as well if that format is required by your instructor.
Source, data and measured evidence have at least three local Git commits. The
initial commit preserves the Champion rule before the completed evaluation.

## Final documents and archive

The final record contains 23 pages covering the mini-project, all fourteen labs,
both quizzes, the assessment table and limitations. The separate quiz sheet has
two pages. Each has an editable DOCX and a PDF. The PDFs are the visually reviewed
submission copies; DOCX pagination can vary between office applications.

`Submission.zip` contains the documents, executable source, data, evidence,
original references, tests and a SHA-256 file manifest. Its checksum is in
`Submission.sha256`. Virtual environments, IDE files, caches, raw tracking stores,
model weights and document previews are excluded. See `SUBMISSION_GUIDE.md` for
the last steps before uploading.

To rebuild after changing student.json or saved evidence:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-evidence.txt
.\.venv\Scripts\python.exe build_documents.py
.\.venv\Scripts\python.exe render_documents.py
# Review every page image under outputs/final-qa/ before packaging.
.\.venv\Scripts\python.exe make_submission.py
```

The document renderer maps the Word content to local HTML and uses an isolated
headless Microsoft Edge process to export PDFs. It checks that every source text
block survives PDF export, then creates page PNGs for visual review. It does not
upload document content. Live model verification and the isolated fresh-run check
passed on 13 November 2026; all 17 tests and all fourteen lab checks passed.

## Audit

See `ASSIGNMENT_AUDIT.md` for the requirement-by-requirement review, fixes and
remaining submission actions. Run `audit_labs.py` to check all fourteen recorded
lab exercises against recomputed results and the live tracking server.

References: [MLflow PythonModel](https://mlflow.org/docs/latest/ml/model/python_model/),
[MLflow registry](https://mlflow.org/docs/latest/ml/model-registry/), and each
candidate's Hugging Face model card. The missing metrics table in the brief is
handled with standard binary classification metrics, averaged over both labels.
