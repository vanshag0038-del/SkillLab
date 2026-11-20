# Final submission guide

Student: HARSHA BS | USN: 1MS25CI051 | Section: A
Prepared: 13 November 2026

## Files to use

| File | Purpose |
|---|---|
| Completed_Lab_Record.pdf | Visually reviewed 23-page record: project, 14 labs, quizzes and assessment table |
| Completed_Lab_Record.docx | Editable copy for the official course details |
| Completed_Quiz_Answers.pdf | Visually reviewed two-page quiz answer sheet |
| Completed_Quiz_Answers.docx | Editable quiz answer sheet |
| Submission.zip | Clean source, data, evidence, references, tests and documents |
| SUBMISSION_MANIFEST.json | SHA-256 and size of every packaged file except the manifest itself |
| Submission.sha256 | Checksum of the completed ZIP; kept alongside the ZIP |

Both quizzes are also included in the main lab record. Marks and instructor
signature/date are left for the instructor. Course title and course code are
blank because official values were not supplied.

## Before uploading

1. Fill the official course title/code. For regenerated copies, edit student.json,
   run build_documents.py and render_documents.py, and review the page images.
2. Add a native terminal screenshot showing successful prediction. The existing
   prediction_terminal.png is clearly labeled as a rendering of the actual saved
   transcript. Native screenshot capture is unavailable in this session.
3. Rerun make_submission.py after edits so the ZIP and manifest include them.
4. Push the final local revision to the existing repository and grant instructor
   access if it remains private. No push was performed during this review.
5. Submit the repository link at the LMS assignment and save the receipt. The
   assignment specifies a GitHub repository link; the ZIP is a supporting archive.

Repository: https://github.com/harsha31bs07-blip/1MS25CI051-mlflow-champion-challenger

## What was checked

17 automated tests pass. Both original MLflow models reproduce all 60 saved
predictions; the Champion alias reproduces the eight new predictions. All 14 lab
audits pass. An isolated fresh evaluation using the original exported local model
snapshots reproduced both prediction sequences and the Champion decision, and
confirmed complete candidate artifacts. First-time Hub download availability was
not retested. All 25 final PDF pages were visually reviewed.

## Recreate on another computer

Extract the archive and follow README.md in order. The assignment excludes
tracking databases and model weights from submission, so a fresh machine must
install dependencies and execute evaluate.py, register_champion.py,
predict_from_mlflow.py and run_labs.py before live verification. Existing run IDs
in saved evidence refer to the original local tracking store.

The ZIP excludes .venv, .idea, .git, caches, model weights, mlruns, mlartifacts,
databases and preview images. Keep the existing local tracking database and model
artifacts on this computer for the viva. The manifest lets the instructor verify
the included files without needing the local runtime.
