# ChargeCast

ChargeCast 是一个“交通流预测 + 充电负荷预测”的端到端工程，分为两段：

1. `flow` 阶段：基于路网拓扑做多节点交通流预测（外生/内生节点解耦建模）
2. `charge` 阶段：结合流量与充电数据，做站点级比率/功率建模，并评估总充电功率

项目默认提供示例数据，用于验证代码链路与字段结构；真实业务数据可直接按同样格式替换。

## 一、项目全景

### 1) 方法流程

```text
flow train/test + adj
    └─(节点对齐 adj ∩ train ∩ test)
        └─ 两阶段流量训练
           ├─ Stage-1: 外生节点单点 LSTM
           └─ Stage-2: 内生节点传播/分配 + 冻结外生模型
                └─ 得到 gate flow 预测
                    └─ 充电数据集构建（ratio / avg_power）
                        └─ 站点级 Beta + LogNormal 训练
                            └─ 端到端总功率评估 (MAE / RMSE)
```

### 2) 目录结构

```text
.
├─ flowConfig/
│  └─ config.py                         # 全局配置
├─ src/
│  ├─ data/
│  │  ├─ flow_data.py                   # 流量数据读取与序列构造
│  │  └─ alignment.py                   # canonical 节点对齐
│  └─ models/
│     ├─ flow_predictor.py              # 核心主模型
│     ├─ lstm.py                        # 外生节点 LSTM
│     ├─ allocation.py                  # 分配模块
│     ├─ propagation.py                 # 传播模块
│     └─ topology.py                    # 拓扑处理
├─ flowScripts/
│  ├─ train.py                          # 两阶段训练入口
│  ├─ test.py                           # 单步测试
│  ├─ visualize.py                      # 预测可视化
│  ├─ predict.py                        # 推理工具函数
│  ├─ recursive_test.py                 # 旧版递归测试脚本（可选）
│  └─ lstm_baseline.py                  # 基线脚本（可选）
├─ flowEvaluation/
│  ├─ multi_step_evaluate.py            # 内生递归多步评估
│  └─ visualize_endogenous.py           # 内生递归可视化
├─ chargePrediction/
│  ├─ prepare_charge_data.py            # charge 节点对齐过滤（可原地覆盖）
│  ├─ prepare_charge_dataset.py         # 构建 charge 训练/测试文件
│  ├─ train_station_models.py           # 站点级模型训练
│  ├─ test_station_models.py            # 端到端总功率评估
│  └─ visualize_total_power.py          # 端到端可视化
├─ data/
│  ├─ flow/                             # flow 数据 (adj/train/test)
│  └─ charge/                           # charge 数据 (charge/train/test)
└─ models/
   └─ best_model.pt                     # flow 阶段 Stage-2 默认 checkpoint
```

## 二、环境准备

### 1) Python 与依赖

- Python >= 3.10
- 主要依赖：`torch`, `numpy`, `pandas`, `matplotlib`

```bash
pip install torch numpy pandas matplotlib
```

### 2) 运行入口

所有脚本默认在仓库根目录执行，例如：

```bash
python flowScripts/train.py
```

## 三、数据格式要求

### 1) Flow 数据 (`data/flow/train.csv`, `data/flow/test.csv`)

必需列：

- `date`
- `slice_start`
- `hour_code`
- `week_code`
- `station_id`
- `nev_flow`

### 2) 拓扑数据 (`data/flow/adj.csv`)

- 行索引与列名均为节点 ID
- `value > 0` 表示有方向边（source -> target）

### 3) Charge 原始数据 (`data/charge/charge.csv`)

端到端测试最少需要这些列：

- `node_id`
- `date`
- `hour_code`
- `week_code`
- `nev_flow`（充电车流）
- `power`（总充电功率）

若不存在 `station_id`，脚本会自动用 `node_id` 补齐。

### 4) Charge 建模最终文件 (`data/charge/train.csv`, `data/charge/test.csv`)

由 `prepare_charge_dataset.py` 生成，字段固定为：

- `hour_code`
- `week_code`
- `station_id`
- `node_id`
- `charge_flow_ratio` = `charge_nev_flow / gate_nev_flow`
- `charge_power` = `charge_power_total / charge_nev_flow`（平均功率）

## 四、核心模块说明

### 1) Flow 阶段模型

- 外生节点：`MultiNodeLSTMPredictor`
- 内生节点：
  - `FourierAllocation` 负责上游出边归一化分配
  - `BayesianPropagation` 负责时序传播概率
- 主模型：`FlowPredictor`
  - 支持 `scheduled sampling`
  - 支持内生节点高斯补偿 `ENDOGENOUS_GAUSSIAN`
  - 可选物理空间非负约束 `OUTPUT_CONSTRAINTS.physical_non_negative`

### 2) 节点对齐策略

默认策略在 `flowConfig/config.py`：

- `NODE_ALIGNMENT.enabled = True`
- canonical 节点集合 = `adj ∩ train ∩ test`

Flow 训练、评估、Charge 对齐都复用该策略，避免节点集合不一致。

### 3) Charge 阶段模型

对每个 `node_id` 独立训练两类模型：

1. 比率模型：Beta 分布，预测 `charge_flow_ratio`
2. 平均功率模型：LogNormal 分布，预测 `charge_power`

特征为 `hour_code`、`week_code` 的一/二阶傅里叶基。

## 五、推荐运行顺序（完整复现）

```bash
# 0) 可选：先过滤 charge.csv 到 canonical 节点（默认会备份并原地覆盖）
python chargePrediction/prepare_charge_data.py

# 1) 训练 flow 两阶段模型
python flowScripts/train.py

# 2) flow 单步测试与可视化
python flowScripts/test.py
python flowScripts/visualize.py

# 3) flow 内生递归多步评估
python flowEvaluation/multi_step_evaluate.py --steps 6
python flowEvaluation/visualize_endogenous.py --steps 6 --max-nodes 16

# 4) 生成 charge 建模数据（默认优先使用 flow 模型预测流量）
python chargePrediction/prepare_charge_dataset.py

# 5) 训练站点级 charge 模型
python chargePrediction/train_station_models.py

# 6) 端到端总功率评估 + 可视化
python chargePrediction/test_station_models.py
python chargePrediction/visualize_total_power.py
```

## 六、主要输出文件

### 1) Flow 阶段

- `models/exogenous/*.pt`：Stage-1 外生节点 checkpoint
- `models/best_model.pt`：Stage-2 主模型 checkpoint
- `models/visualization.png`：flow 预测可视化

### 2) Flow 递归评估

- `flowEvaluation/results/recursive_endo_metrics.csv`
- `flowEvaluation/results/recursive_endo_metrics.png`
- `flowEvaluation/results/endogenous_recursive_visualization.png`

### 3) Charge 数据准备

- `chargePrediction/data/charge_model_all.csv`
- `chargePrediction/data/charge_model_train.csv`
- `chargePrediction/data/charge_model_test.csv`
- `chargePrediction/data/dataset_report.json`
- `data/charge/train.csv`（最终训练输入）
- `data/charge/test.csv`（最终测试输入）

### 4) Charge 训练/评估

- `chargePrediction/models/ratio/<node_id>.pt`
- `chargePrediction/models/power/<node_id>.pt`
- `chargePrediction/models/training_summary.csv`
- `chargePrediction/models/training_report.json`
- `chargePrediction/models/e2e_test_detail.csv`
- `chargePrediction/models/e2e_test_node_metrics.csv`
- `chargePrediction/models/e2e_test_report.json`
- `chargePrediction/models/e2e_total_power_visualization.png`

## 七、常用参数

### 1) `flowEvaluation/multi_step_evaluate.py`

- `--steps`：每个 block 递归预测步数（默认 `6`）
- `--output-dir`：指标输出目录

### 2) `chargePrediction/prepare_charge_dataset.py`

- `--use-canonical-filter`：是否按 canonical 节点过滤 charge 数据
- `--no-model-pred`：禁用 flow 模型预测流量（退回真值）
- `--strict-model-pred`：预测流量失败时直接报错
- `--final-train-path` / `--final-test-path`：最终导出路径

### 3) `chargePrediction/train_station_models.py`

- `--ratio-epochs`, `--power-epochs`
- `--ratio-lr`, `--power-lr`
- `--min-samples`：每站点最少样本数

### 4) `chargePrediction/test_station_models.py`

- `--test-ratio`：末尾时间切片比例（默认 `0.2`）
- `--station-model-dir`：站点模型目录
- `--use-canonical-filter`

## 八、关键配置（`flowConfig/config.py`）

- 数据与设备：`DATA_DIR`, `MODEL_DIR`, `SEQ_LEN`, `PRED_LEN`, `DEVICE`
- 节点对齐：`NODE_ALIGNMENT`
- Stage-1：`EXOGENOUS_TRAIN`
- Stage-2：`ENDOGENOUS_TRAIN`
- 训练策略：`SCHEDULED_SAMPLING`
- 输出约束：`OUTPUT_CONSTRAINTS`
- 内生补偿：`ENDOGENOUS_GAUSSIAN`

## 九、常见问题

1. 报错 `Checkpoint not found: models/best_model.pt`
   先运行 `python flowScripts/train.py` 生成 Stage-2 checkpoint。

2. 报错 `No canonical nodes found` 或节点缺失
   检查 `adj.csv`、`train.csv`、`test.csv` 三者节点交集是否为空，必要时关闭/调整 `NODE_ALIGNMENT`。

3. 端到端评估节点大量被跳过
   一般是某些站点缺少 `ratio/power` checkpoint，先确认 `train_station_models.py` 的 `--min-samples` 设置与训练数据覆盖。

4. 指标异常或样本很少
   重点检查 `charge.csv` 的 `nev_flow/power` 是否存在大量零值或缺失值，以及时间列是否可正确解析。
