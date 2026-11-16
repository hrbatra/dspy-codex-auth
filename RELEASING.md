# Releasing to PyPI

The PyPI package for this repo is:

https://pypi.org/project/dspy-codex-auth/

The GitHub repo is:

https://github.com/hrbatra/dspy-codex-auth

PyPI releases are immutable. Every package update needs a new version number,
even if the change is only README/docs.

## Dependency on openai-codex-auth

This package depends on the published
[`openai-codex-auth`](https://pypi.org/project/openai-codex-auth/) package for
Codex CLI credentials, HTTP/WebSocket transports, and streaming reconstruction.
This repository owns DSPy integration; the core client also works independently
of DSPy. No Pi plugin or credential store is required.

The 0.3 series requires DSPy 3.5+. `pyproject.toml` and `uv.lock` resolve the auth package from PyPI, with no local
path source. When a change here needs a newer auth API, release
`openai-codex-auth` first, update the dependency with `uv add`, and commit
`pyproject.toml` and `uv.lock`. Verify with `uv sync --locked --dev` and
`scripts/check` before releasing.

## Local Release

Use this path when publishing from your machine with a PyPI token in
`~/.pypirc` or `TWINE_PASSWORD`.

1. Start from a clean checkout.

```bash
cd /Users/rsika/dev/dspy-codex-auth
git status --short
git pull --ff-only origin main
```

2. Make the change, update `CHANGELOG.md`, and bump the version with uv.

```bash
uv version --bump patch
```

Use `--bump minor` or `--bump major` for larger compatibility changes. The
DSPy 3.5 migration is the minor release 0.3.0. Do not edit `pyproject.toml`
manually to change the version or dependencies.

3. Verify the source and an isolated installation of its wheel.

```bash
uv sync --locked --dev
scripts/check
```

The check runs lint, tests, a temporary distribution build, and the offline
sync/async smoke test. This exercises the installed adapter and shared client;
it does not require live subscription credentials.

4. Build and verify the release distributions.

```bash
release_version=$(uv version --short)
uv build --no-sources
uv run --with twine python -m twine check \
  "dist/dspy_codex_auth-$release_version-py3-none-any.whl" \
  "dist/dspy_codex_auth-$release_version.tar.gz"
```

Use the exact versioned paths when uploading so older files in `dist/` cannot
be published accidentally.

5. Commit the reviewed change and version bump, then push to GitHub.

```bash
git diff --check
git status --short
# Stage only the files intended for this release.
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "Release $release_version"
git push origin main
```

Ensure all implementation, test, and documentation changes are committed too.
Check that GitHub CI passes on the pushed commit before publishing.

6. Upload to PyPI using **one** publishing path.

For a local token-based upload:

```bash
uv run --with twine python -m twine upload \
  "dist/dspy_codex_auth-$release_version-py3-none-any.whl" \
  "dist/dspy_codex_auth-$release_version.tar.gz"
```

Alternatively, use the tag-triggered Trusted Publishing workflow below. Do not
also upload the same version locally: PyPI releases are immutable, and a second
upload will fail.

7. Verify a fresh install from PyPI.

```bash
uv run --isolated --no-project --refresh \
  --with "dspy-codex-auth==$release_version" python tests/smoke_test.py
```

This resolves the published package rather than the working tree. Confirm the
same version is listed on PyPI, and include a notebook-kernel restart in the
upgrade instructions for users.

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
committed, pushed, and CI passes:

```bash
release_version=$(uv version --short)
git tag "v$release_version"
git push origin "v$release_version"
```

Verify the Publish workflow succeeds and then run the fresh PyPI smoke test
above. A repository workflow alone does not prove the corresponding Trusted
Publisher is registered in PyPI; a token-based local upload is the other
documented release path.

Do not reuse tags or PyPI versions. If a release fails after uploading any file,
bump to a new version before trying again.
