# Contributing to LumenMorph

Contributions are welcome when they improve reproducibility, usability or the
scientific usefulness of the generator without overstating what it validates.

## Before opening a pull request

1. Open an issue for substantial features or changes to generated data schema.
2. Keep configurations, seeds and expected outputs explicit.
3. Do not silently alter a frozen benchmark profile or reinterpret synthetic
   truth as clinical validation.
4. Update the relevant user and scientific documentation.
5. Add focused tests when behavior changes.

Run the applicable checks from the repository root:

```bash
backend/.venv/bin/python -m pytest -q
npm --prefix frontend run lint
npm --prefix frontend test
npm --prefix frontend run build
```

For a benchmark or mechanics change, also record the exact generator revision,
configuration, seed, dependency environment, QA output and interpretation.
See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Pull requests

Describe the user-visible behavior, validation performed, and any changed
scientific claim or limitation. Keep generated datasets and bulky output out of
Git; commit the configuration, source and compact evidence needed to reproduce
them instead.
