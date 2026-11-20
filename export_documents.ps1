# Render the final documents locally through the isolated headless Edge engine.
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'render_documents.py')
exit $LASTEXITCODE
