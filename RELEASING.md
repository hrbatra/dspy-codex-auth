# Releasing to PyPI

The PyPI package for this repo is:

https://pypi.org/project/dspy-codex-auth/

The GitHub repo is:

https://github.com/hrbatra/dspy-codex-auth

PyPI releases are immutable. Every package update needs a new version number,
even if the change is only README/docs.

## Local Release

Use this path when publishing from your machine with a PyPI token in
`~/.pypirc` or `TWINE_PASSWORD`.

1. Start from a clean checkout.

```bash
cd /Users/rsika/dev/dspy-codex-auth
git status --short
git pull --ff-only origin main
```

2. Make and commit the change.

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
git add .
git commit -m "Describe the change"
git push origin main
```

3. Bump the version.

```bash
uv version --bump patch
```

Use `--bump minor` or `--bump major` only for larger compatibility changes.

4. Build and verify the distributions.

```bash
rm -rf dist
uv build --no-sources
uv run --with twine python -m twine check dist/*
```

5. Commit and push the version bump.

```bash
version=$(uv version --short)
git add pyproject.toml uv.lock
git commit -m "Release $version"
git push origin main
```

6. Upload to PyPI.

```bash
uv run --with twine python -m twine upload dist/*
```

7. Verify PyPI and a fresh install.

```bash
curl -s https://pypi.org/pypi/dspy-codex-auth/json | python3 -m json.tool | head

tmpdir=$(mktemp -d /tmp/dspy-codex-auth-pypi.XXXXXX)
cd "$tmpdir"
uv init --bare
uv add --refresh dspy-codex-auth
uv run python - <<'PY'
import importlib.metadata
import importlib.util
import dspy_codex_auth

print(importlib.metadata.version("dspy-codex-auth"))
print(importlib.util.find_spec("dspy_lm_auth"))
print(dspy_codex_auth.LM.__module__)
PY
rm -rf "$tmpdir"
```

`find_spec("dspy_lm_auth")` should print `None`; `dspy-codex-auth` should not
install `dspy-lm-auth` as a runtime dependency.

## If PyPI or uv Shows the Previous Version

Right after upload, PyPI and uv can briefly serve cached index data. Force a
fresh resolution with:

```bash
uv add --refresh dspy-codex-auth
```

For an existing repo:

```bash
uv lock --refresh-package dspy-codex-auth --upgrade-package dspy-codex-auth
uv sync
```

The version-specific PyPI URL is the quickest way to confirm the upload exists:

```bash
open https://pypi.org/project/dspy-codex-auth/
```

## GitHub Actions Publishing

The repo also has `.github/workflows/publish.yml`, which can publish tag pushes
through PyPI Trusted Publishing if PyPI is configured for:

- PyPI project: `dspy-codex-auth`
- Owner: `hrbatra`
- Repository: `dspy-codex-auth`
- Workflow: `publish.yml`
- Environment: `pypi`

Once that is configured, release by pushing a tag after the version bump is
committed:

```bash
version=$(uv version --short)
git tag "v$version"
git push origin "v$version"
```

Do not reuse tags or PyPI versions. If a release fails after uploading any file,
bump to a new version before trying again.
