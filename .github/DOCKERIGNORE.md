# .dockerignore is read from the build-context ROOT, not from docker/

## The problem

`docker/.dockerignore` **has no effect**. The BuildKit / Docker daemon always
resolves `.dockerignore` relative to the *build context root*. The agri build
uses the repository root as the context:

```bash
docker build --file docker/Dockerfile --target production -t agri-platform:latest .
#            ^^^^^^^^^^^^^^^^^^^^^ explicit Dockerfile      ^ context root = repo root
```

Therefore the daemon looks for `<repo-root>/.dockerignore` and never reads
`docker/.dockerignore`.

## Why it is not fixed here

This change-set's write scope is `docker/` and `.github/` only, so the
repository-root `.dockerignore` could not be created.

## What was done instead

1. `docker/.dockerignore` exists as a **template** with the full ignore list.
2. The same list is reproduced in the header comment of `docker/Dockerfile`.
3. `docker/README.md` documents the caveat in its "Building" section.
4. `docker/Dockerfile` was written so the build stays **correct** without the
   file: every `COPY` is explicit and narrow (`pyproject.toml`, `README.md`,
   `src/`, `configs/`), so `data/`, `models/`, `.env` and `.venv` can never be
   baked into a layer even when they are sent to the daemon.

Without the root `.dockerignore` the build is merely **slower**: the working
tree (raw imagery under `data/raw/`, the local `.venv/`) is uploaded to the
daemon on every build.

## Action required (repository owner)

Copy the template to the repository root and commit it:

```bash
cp docker/.dockerignore .dockerignore
```

Expected content:

```gitignore
# --- vcs / ci / tooling ------------------------------------------------------
.git/
.gitignore
.github/
.vscode/
.idea/

# --- python virtualenvs & caches --------------------------------------------
.venv/
venv/
**/__pycache__/
**/*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
.coverage.*
coverage*.xml
htmlcov/
.ipynb_checkpoints/

# --- data, models, reports: mounted at run time, never baked -----------------
data/
models/
reports/
mlruns/
mlartifacts/

# --- docs / dev-only trees ---------------------------------------------------
docs/
notebooks/
scripts/
tests/
node_modules/
.vite/
*.tsbuildinfo

# --- secrets: never send them to the daemon ---------------------------------
.env
.env.*
!.env.example
*.pem
*.key

# --- os / editor noise ------------------------------------------------------
.DS_Store
Thumbs.db
desktop.ini
*.swp
```

## Verification once it is in place

```bash
docker build --file docker/Dockerfile --target production --progress=plain -t agri-platform:latest . 2>&1 | head -20
# "transferring context" should now report a few MB, not gigabytes
```
