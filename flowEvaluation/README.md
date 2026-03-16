# Multi-step Evaluation

This folder evaluates **endogenous recursive multi-step forecasting** and provides
an endogenous-only recursive visualization script.

## Dataset used

Both scripts use `test.csv` (test set).

## Protocol (both scripts)

For each evaluation block:

1. Use real history of length `SEQ_LEN` as model input.
2. Recursively predict `--steps` steps.
3. During recursion:
   - endogenous nodes use predicted values (autoregressive),
   - exogenous nodes are forced with ground-truth values each step.
4. After one block is done, reset to real data and start the next block.

## Scripts

- `multi_step_evaluate.py` (metrics)
- `visualize_endogenous.py` (plots)

## Usage

Recursive endogenous multi-step metrics:

```bash
python flowEvaluation/multi_step_evaluate.py --steps 6
```

Recursive endogenous multi-step visualization:

```bash
python flowEvaluation/visualize_endogenous.py --steps 6 --max-nodes 16
```

## Outputs

- `flowEvaluation/results/recursive_endo_metrics.csv`
- `flowEvaluation/results/recursive_endo_metrics.png`
- `flowEvaluation/results/endogenous_recursive_visualization.png`

CSV columns include per-step MAE/RMSE for endogenous nodes (plus exogenous/all for reference).
