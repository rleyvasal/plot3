# Releasing plot3

A release is one version bump and one tag. GitHub Actions does the rest:
it runs the tests, builds the wheel and sdist, uploads them to PyPI, and
creates a GitHub release with the files attached.

## Each release

1. Set the new version in `plot3/__version__.py` (the only place it lives)
   and add a section to `CHANGELOG.md`.
2. Commit, tag, and push:

   ```bash
   git commit -am "Release 0.4.1"
   git tag v0.4.1
   git push origin main v0.4.1
   ```

The tag must equal `v` + `__version__`, or the workflow stops before
uploading. PyPI never accepts the same version twice, so a mistake means a
new version number, not a re-upload.

Watch the run under the repository's **Actions** tab, or with
`gh run watch`.

## One-time setup (already done once per project)

PyPI trusted publishing lets the workflow upload without any token or
password stored in GitHub or on your computer.

1. Sign in at <https://pypi.org> (with two-factor authentication on).
2. Go to **Your projects → Publishing → Add a new pending publisher** and enter:
   - PyPI project name: `plot3`
   - Owner: `rleyvasal`
   - Repository name: `plot3`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. In the GitHub repository, **Settings → Environments → New environment**
   named `pypi`. Optionally add yourself as a required reviewer, so each
   upload waits for one click of approval.

The first tagged release creates the project on PyPI; after that the
pending publisher becomes a normal one.

## Checking a build locally

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```
