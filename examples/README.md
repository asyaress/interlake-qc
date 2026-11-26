# New-user example (synthetic data)

This demo uses **synthetic** WSE series in `data/example/synthetic_two_lakes.csv` (not real SWOT).

## Minimal columns

`time`, `lake`, `wse`, `quality_f` (optional `pass`).

## Run

```bash
interlake-qc run --config configs/example_pair.yaml
# or:
python -c "from interlake_qc.cli_example import main; main()"
python scripts/run_example.py
```

Outputs land in `outputs/example/`: paired passes, D2 scores, and `flagged_passes.csv`.
