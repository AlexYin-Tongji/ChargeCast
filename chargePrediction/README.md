# Charge Prediction Workspace

This folder is used for charge forecasting tasks.

## 1) Data Preparation

### 1.1 Node Alignment (in-place filter)

- Script: `prepare_charge_data.py`
- Purpose: keep only `node_id` in flow canonical nodes (`adj ∩ train ∩ test`).

### 1.2 Charge modeling dataset

- Script: `prepare_charge_dataset.py`
- Final outputs:
  - `data/charge/train.csv`
  - `data/charge/test.csv`

Important:

- `charge_power` in final files is now **average power**:
  - `charge_power = total_power / charge_nev_flow`
- `charge_flow_ratio = charge_nev_flow / gate_nev_flow`

## 2) Per-station Model Training

- Script: `train_station_models.py`
- Each `node_id` is trained independently.
- No validation/test split inside this script; it uses training file rows directly.

Models:

1. Ratio model (`charge_flow_ratio`): Beta likelihood, learn `alpha`, `beta`.
2. Power model (`charge_power`): LogNormal likelihood, learn `mu`, `sigma`.
3. Feature basis: hour/week first and second harmonics:
   - `sin/cos(2πk*hour/24)`, `sin/cos(2πk*week/7)`, `k in {1,2}`.

Loss:

- Ratio: `L_ratio = -mean(log Beta(y | alpha, beta))`
- Power: `L_power = -mean(log LogNormal(y | mu, sigma))`

Outputs:

- `chargePrediction/models/ratio/<node_id>.pt`
- `chargePrediction/models/power/<node_id>.pt`
- `chargePrediction/models/training_summary.csv`
- `chargePrediction/models/training_report.json`

Run:

```bash
python chargePrediction/train_station_models.py
```

## 3) End-to-end Test (Final Metric)

- Script: `test_station_models.py`
- Final metric target: **total charge power**
- Test split: **last 20% by time**

End-to-end inference in test:

1. Use traffic flow model checkpoint to predict gate flow.
2. Use station Beta model to predict charge flow ratio.
3. Use station LogNormal model to predict average charge power.
4. Compose total power prediction:
   - `pred_total_power = pred_gate_flow * pred_ratio * pred_avg_power`

Evaluation metrics:

- `MAE(total_power)`
- `RMSE(total_power)`

Outputs:

- row-level detail: `chargePrediction/models/e2e_test_detail.csv`
- node-level summary: `chargePrediction/models/e2e_test_node_metrics.csv`
- aggregate report: `chargePrediction/models/e2e_test_report.json`

Run:

```bash
python chargePrediction/test_station_models.py
```

## 4) Visualization

- Script: `visualize_total_power.py`
- Function: similar to flow visualization, plots multi-station curves of:
  - true total power
  - predicted total power

Output:

- `chargePrediction/models/e2e_total_power_visualization.png`

Run:

```bash
python chargePrediction/visualize_total_power.py
```
