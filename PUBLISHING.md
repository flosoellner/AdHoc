# Publication checklist

Use this before releasing code (e.g. thesis supplement, Zenodo, or PyPI).

## Code and repo

- [ ] Remove or redact any local paths, API keys, or personal data.
- [ ] Ensure `.gitignore` is complete; no `__pycache__/`, `*.pt`, `cache_*/`, or large outputs committed.
- [ ] Run from a clean clone: `git clone ... && cd AdHoc && pip install -e .` and run one notebook to confirm.
- [ ] Optional: run a linter (e.g. `ruff check .`) and fix critical issues.

## Dependency and install

- [ ] `requirements.txt` and `pyproject.toml` list all runtime deps; versions are pinned or lower-bounded as you prefer.
- [ ] `pip install -r requirements.txt` (or `pip install -e .`) succeeds on a fresh environment (e.g. Python 3.10+).

## Documentation

- [ ] `README.md` describes the project, setup, and how to run experiments.
- [ ] Docstrings on public functions/modules are clear enough for readers.
- [ ] Add a `LICENSE` file (e.g. MIT) if you publish the repo.

## Optional (Zenodo / PyPI)

- [ ] Tag a release: `git tag v0.1.0`.
- [ ] For Zenodo: connect the repo and create a release; Zenodo will mint a DOI.
- [ ] For PyPI: `python -m build && twine upload dist/*` (after configuring credentials). Prefer Test PyPI first.

## Cleanup before tagging

```bash
# From repo root
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null
# Ensure no large or sensitive files are staged
git status
```
