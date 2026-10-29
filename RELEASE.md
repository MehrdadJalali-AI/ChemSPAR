# Releasing ChemSPAR

## One-time setup

1. **PyPI Trusted Publishing.** Sign in to pypi.org, then *Your projects → Publishing → Add a new pending
   publisher*:
   - PyPI project name: `chemspar`
   - Owner: `MehrdadJalali-AI`, repository: `ChemSPAR`
   - Workflow name: `release.yml`, environment: `pypi`

   (The pending publisher reserves the name only when the first upload succeeds.)
2. **GitHub environment.** In the repository: *Settings → Environments → New environment* `pypi`, and add
   yourself as a **required reviewer**. Pushing the tag builds and tests immediately, but the upload then waits
   for your approval, which is what lets you follow the checklist order below (PyPI last).
3. **Zenodo.** Sign in to zenodo.org with GitHub, open *GitHub* in the account menu and switch on
   `MehrdadJalali-AI/ChemSPAR`. Zenodo then archives every GitHub release and issues a software DOI.
4. For the TestPyPI dry run, create an API token on test.pypi.org (*Account settings → API tokens*).

## Dry run on TestPyPI (from a clean checkout of the release commit)

```bash
python -m pip install --upgrade build twine
rm -rf dist && python -m build
twine check --strict dist/*

# upload (username __token__, password = the TestPyPI token)
twine upload --repository testpypi dist/*

# install from TestPyPI into a fresh environment; dependencies come from the real PyPI
python -m venv /tmp/chemspar-dryrun && source /tmp/chemspar-dryrun/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ chemspar==1.0.0
chemspar --version
chemspar policies

# run the test suite against the installed package (not the source tree)
mkdir /tmp/chemspar-dryrun-tests && cp -r tests paper /tmp/chemspar-dryrun-tests/ && cd /tmp/chemspar-dryrun-tests
pip install pytest && pytest -q tests
deactivate
```

A version number can be uploaded to TestPyPI only once. For a second dry run, use a pre-release version such as
`1.0.0rc2` in `chemspar/__init__.py` (do not tag it).

## Release checklist

- [ ] All placeholders filled (`grep -rn "\[ARTICLE DOI\]\|\[YEAR\]\|\[ORCID\|\[GIVEN NAME\|\[ZENODO RECORD" .`).
- [ ] `CHANGELOG.md`: replace "unreleased" with the date.
- [ ] CI green on `main`, including the `test-paper-stack` job (exact golden reproduction).
- [ ] TestPyPI dry run done.

1. **Tag** the release commit: `git tag -a v1.0.0 -m "chemspar 1.0.0" && git push origin v1.0.0`.
   The tag must match `__version__`; the release workflow checks this.
2. **Create the GitHub release** from tag `v1.0.0` (notes from `CHANGELOG.md`).
   Pushing the tag starts `release.yml`: build, `twine check`, test of the installed wheel, then the `publish`
   job, which waits for the `pypi` environment if you set a reviewer.
3. **Zenodo** archives the GitHub release and issues a software DOI (a version DOI and a concept DOI that always
   resolves to the latest version).
4. **Add the software DOI** to `README.md` (citation section) and `CITATION.cff` (`doi:` at the top level, and an
   identifiers entry), commit to `main`. These edits go into the next release; the archived 1.0.0 keeps the
   version without them.
5. **Publish to PyPI**: approve the waiting `publish` job of the tag's workflow run. The uploaded package is the
   one built from the tag, so its PyPI page shows the README with the article DOI but without the software DOI.
   Check https://pypi.org/project/chemspar/ and `pip install chemspar==1.0.0` in a fresh environment.
6. **Announce the paper and the package together**, linking the article DOI, the PyPI page and the software DOI.
