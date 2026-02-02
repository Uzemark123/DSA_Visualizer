# Running tests

From the repo root:

```bash
python -m pytest backend/app/tests
```

Notes:
- Route tests build a tiny sqlite DB fixture automatically.
- If you want verbose output:

```bash
python -m pytest backend/app/tests -vv
```
