"""Build and verify a clean submission archive without local runtime state."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'Submission.zip'
MANIFEST = ROOT / 'SUBMISSION_MANIFEST.json'
ROOT_FILES = [
    '.gitattributes', '.gitignore', 'pytest.ini', 'README.md', 'ASSIGNMENT_AUDIT.md',
    'SUBMISSION_GUIDE.md', 'student.json', 'quiz_answers.json', 'submission_status.json',
    'requirements.txt', 'requirements-lock.txt', 'requirements-evidence.txt',
    'config.py', 'hf_model.py', 'evaluate.py', 'register_champion.py',
    'predict_from_mlflow.py', 'run_labs.py', 'verify_submission.py', 'audit_labs.py',
    'capture_evidence.py', 'build_documents.py', 'build_record.cjs',
    'export_documents.ps1', 'render_documents.py', 'make_submission.py',
    'Completed_Lab_Record.docx', 'Completed_Lab_Record.pdf',
    'Completed_Quiz_Answers.docx', 'Completed_Quiz_Answers.pdf',
    'MLflow_Lab_Book_with_Rubrics.docx', 'Day_1_MCQs_with_Answer_Key.docx',
    'Day_2_MCQs_with_Answer_Key.docx',
]
FOLDERS = {
    'data': {'.csv', '.md'}, 'labs': {'.py', '.md', ''},
    'outputs': {'.csv', '.json', '.md', '.txt', '.png', '.xlsx'},
    'tests': {'.py'},
}
SCREENSHOTS = ['README.md', 'mlflow_compare.png', 'registry_aliases.png', 'prediction_terminal.png']
EXCLUDED_PARTS = {
    '.git', '.idea', '.venv', '__pycache__', '.pytest_cache', '.artifact-tools',
    '.cache', 'mlruns', 'mlartifacts', 'models', 'document-preview', 'final-qa',
    'package-check',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_files():
    files = [ROOT / name for name in ROOT_FILES]
    files.extend(ROOT.glob('Assignment*.docx'))
    for folder, extensions in FOLDERS.items():
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and not (set(path.relative_to(ROOT).parts) & EXCLUDED_PARTS):
                if path.suffix in extensions:
                    files.append(path)
    files.extend(ROOT / 'screenshots' / name for name in SCREENSHOTS)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def main():
    files = selected_files()
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Required submission files missing: {missing}')
    for report in ['verification.json', 'fresh_run_check.json', 'document_qa.json']:
        data = json.loads((ROOT / 'outputs' / report).read_text(encoding='utf-8'))
        if data.get('status') != 'passed':
            raise ValueError(f'{report} must pass before packaging')
    document_qa = json.loads((ROOT / 'outputs/document_qa.json').read_text())
    for name, details in document_qa['documents'].items():
        if sha256(ROOT / name) != details['sha256']:
            raise ValueError(f'{name} changed after document QA')
    manifest = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'student_usn': json.loads((ROOT / 'student.json').read_text())['usn'],
        'archive': ARCHIVE.name,
        'scope': 'Source, references, measured evidence and final documents; no tracking store or model weights',
        'files': [{'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)} for path in files],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary = ARCHIVE.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=9) as package:
        for path in [*files, MANIFEST]:
            package.write(path, path.relative_to(ROOT).as_posix())
    with ZipFile(temporary) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(files) + 1
        assert not any(set(Path(name).parts) & EXCLUDED_PARTS for name in package.namelist())
        assert not any(name.endswith(('.db', '.bin', '.safetensors', '.pyc', '.zip')) for name in package.namelist())
        for item in manifest['files']:
            assert hashlib.sha256(package.read(item['path'])).hexdigest() == item['sha256']
    temporary.replace(ARCHIVE)
    checksum = f'{sha256(ARCHIVE)}  {ARCHIVE.name}\n'
    (ROOT / 'Submission.sha256').write_text(checksum, encoding='ascii')
    print(f'Created and verified {ARCHIVE.name}: {len(files) + 1} files, {ARCHIVE.stat().st_size:,} bytes')
    print(checksum.strip())


if __name__ == '__main__':
    main()
