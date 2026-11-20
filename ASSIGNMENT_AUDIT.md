# Final assignment audit

Review date: 13 November 2026. Student: HARSHA BS, 1MS25CI051, Section A.

The local implementation, measured evidence and final documents pass the review.
The submission package is prepared locally. Course details, a native terminal
screenshot, instructor access to the private repository and the LMS upload remain
student actions; no external submission or instructor marks are claimed.

## Scope

Reviewed both original briefs, both supplied quiz keys, all project and lab source,
all tests, dependencies, data, saved outputs, registry/model evidence, screenshots,
Git history and generated documents. Third-party installed packages, cached model
weights and tracking-store internals were treated as local runtime state.

## Validation

| Check | Result |
|---|---|
| Automated suite | 17 passed; one existing MLflow advisory about optional prediction type hints |
| Original model replay | Both MLflow packages reproduce all 60 saved predictions |
| Metrics and timing | Four classification metrics independently recomputed; latency arithmetic matches |
| Registry | Champion/challenger aliases resolve to the exact evaluated run IDs |
| New data | Eight disjoint examples reproduced through the Champion alias |
| Lab evidence | All 14 exercise audits pass, including recomputed regression metrics and model reload |
| Quizzes | All 30 responses match the supplied keys; explanations and student identity present |
| Fresh execution | Isolated SQLite store, new runs, original local model snapshots; predictions and winner reproduced |
| Artifact logging fix | Fresh runs each contain the comparison chart and other required artifacts |
| Documents | 23-page lab record and 2-page quiz sheet; all source text survives PDF export |
| Visual review | Every final PDF page image inspected; no clipping, broken tables or blank pages |
| Package | Explicit file selection, ZIP integrity check and per-file SHA-256 verification |

The fresh-run check used the previously exported local Hugging Face weights to
avoid repeating Hub downloads. Hub download availability was not retested. The
original measurements and aliases were preserved. Fresh-run latencies differ
with system load and were not substituted into the submitted results table.

PDFs were generated from the DOCX content via local HTML and headless Microsoft
Edge, then rasterized for review. The packaged LibreOffice renderer was unavailable
and Word PDF export stalled. DOCX pagination can vary between office applications;
the visually reviewed PDF copies are the submission references.

## Requirement review

- Same 60 saved inputs and references in the same order for both models; balanced
  intended labels and data hash verified. Both Hub IDs and exact revisions saved.
- Required A_sentiment, B_sentiment and compare_sentiment run names, parameters,
  metrics, tags, FINISHED statuses, model packages and artifacts verified.
- Initial Git history preserves the original 0.01 macro-F1 margin and 5x latency
  guardrail. The policy is unchanged. A remains Champion at 0.966630 macro-F1
  against B at 0.794286. The explanation is within 150 to 250 words.
- Both aliases select the correct versions. Inference loads the Champion alias,
  prints its version and model ID, and imports no Hugging Face pipeline.
- Two confusion matrices and a complete metric comparison plot are included.
  Twelve disagreements and six specific qualitative comments are preserved.
- All fourteen labs contain saved transcripts and exercise outputs. Lab01 includes
  six workbook sheets and an explicit dataset/split membership CSV. Lab12 records
  both prescribed Project runs; Lab13 records the three lowest-MAE runs.
- Required repository structure, USN repository name and at least three historical
  commits exist. Saved history is also supplied in outputs/git_history.txt.
- Two screenshots show the actual MLflow UI. The third is explicitly a rendering
  of the executed prediction transcript; it is not a native console screenshot.

## Cleanup and fixes in this final review

- Fixed evaluate.py so a fresh comparison logs its chart to both candidate runs.
- Restricted summary artifact logging to files created by the current comparison;
  stale lab stores, document previews and unrelated outputs are no longer swept in.
- Preserved Champion evidence when optional Challenger inference is requested.
- Propagated the active tracking URI to Projects and CLI subprocesses.
- Added regression checks for Challenger evidence isolation and CLI server routing.
- Restricted pytest discovery to tests/ and converted the validation transcript
  to UTF-8 after the extracted-copy check caught it being collected as a doctest.
- Rebuilt documents with student details, blank official course fields, an unfilled
  assessment/signature table, code/output excerpts, plots and both quiz answers.
- Replaced the document-specific npm dependency with the existing Python evidence
  toolchain, keeping build_record.cjs as a compatibility entry point.
- Removed obsolete one-off identity and document-preview helpers. The packager
  excludes environments, IDE files, caches, model weights, raw stores and previews.
- Added a submission guide, reproducible package builder, manifest and checksum.

Earlier fixes remain in place: pinned model revisions, strict input validation,
metric/artifact/alias checks, quiz-key transcription checks and stable line endings.
The original rule, evaluation labels, measured results and historical commits were
not changed. The required AI-use declaration is retained.

## Remaining submission steps

1. Fill the official course title and course code on the cover.
2. Add a native screenshot of predict_from_mlflow.py running successfully; this
   session cannot capture it because the Computer Use native pipe is unavailable.
3. Publish the final local revision to the existing GitHub repository and give the
   instructor access if it stays private. This review did not push any changes.
4. Submit the repository link at the correct LMS assignment and keep the receipt.
5. Review the implementation and quiz explanations for the viva.

The original starter scripts were not supplied. The general regression baseline
is a disclosed synthetic replacement; the delivery and hospital tables reproduce
the brief. Exact equivalence to missing teaching scripts cannot be established.
The supplied answer keys govern quiz questions about those missing scripts.
