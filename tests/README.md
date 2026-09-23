# Test layout

The root of `tests/` contains the model, data-loader, training, and evaluation
tests inherited from the fork. `tests/vessel/` contains the current vessel
graph extraction, patch curation, and physical branch-metric tests.

Run the focused vessel checks with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/vessel
```

The old tests for one-off centerline comparisons, crop-policy experiments,
and relocation were removed with their scripts. The committed versions remain
recoverable in Git history.
