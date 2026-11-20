// Compatibility entry point. Documents now share the Python evidence toolchain.
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const python = process.env.DOCUMENT_PYTHON || path.join(__dirname, '.venv', 'Scripts', 'python.exe');
const result = spawnSync(python, [path.join(__dirname, 'build_documents.py')], { stdio: 'inherit' });
process.exit(result.status ?? 1);
