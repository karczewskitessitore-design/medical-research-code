$ErrorActionPreference = "Stop"

uv venv --python 3.11
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv sync --dev --extra benchmark --no-install-project
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv pip install . --no-deps --reinstall
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

.\.venv\Scripts\python.exe -c "import kidney_ssl, torch, monai; print(kidney_ssl.__version__, torch.__version__, monai.__version__)"
