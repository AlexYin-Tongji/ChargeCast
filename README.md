# ChargeCast

ChargeCast 是一个“交通流预测 + 充电功率预测”的两阶段项目：

- 第一部分：基于拓扑图的交通流预测（区分外生/内生节点）
- 第二部分：基于流量预测结果构建充电数据集，并进行站点级充电建模
- 最终评估：端到端预测总充电功率（MAE/RMSE）

> 说明：仓库中的数据为示例数据，主要用于验证代码链路与数据结构。

## 1. 项目结构

```text
.
├─ flowConfig/                 # 全局配置
│  └─ config.py
├─ src/
│  ├─ data/                    # 数据加载、节点对齐
│  └─ models/                  # 核心模型（LSTM / 分配 / 传播 / 主模型）
├─ flowScripts/                # 交通流训练/测试/推理/可视化
├─ flowEvaluation/             # 内生节点递归多步评估与可视化
├─ chargePrediction/           # 充电预测数据准备、训练、测试、可视化
├─ data/
│  ├─ flow/                    # 交通流数据（adj/train/test）
│  └─ charge/                  # 充电数据（charge/train/test）
└─ models/                     # 流量模型 checkpoint（默认 best_model.pt）
```

## 2. 交通流模型（Flow）

### 2.1 核心思路

1. 节点对齐：统一节点集合为 `adj ∩ train ∩ test`（canonical nodes）
2. 拓扑划分：
   - 外生节点（无上游）
   - 内生节点（有上游）
3. 两阶段训练：
   - Stage-1：每个外生节点单独 LSTM 训练并保存
   - Stage-2：加载并冻结外生模型，训练内生传播与分配模块
4. 前向预测：
   - 外生节点在多步预测中按滚动历史逐步重算
   - 分配权重在同一上游节点的出边上做归一化

### 2.2 关键模块

- `src/models/flow_predictor.py`
  - 外生：`MultiNodeLSTMPredictor`
  - 分配：`FourierAllocation`
  - 传播：`BayesianPropagation`
  - 支持内生高斯补偿（`ENDOGENOUS_GAUSSIAN`）
- `src/data/alignment.py`
  - canonical 节点构建与对齐
- `src/models/topology.py`
  - 外生/内生节点识别、拓扑顺序、上下游查询

### 2.3 常用命令

```bash
# 训练（两阶段）
python flowScripts/train.py

# 单步测试（默认测试集）
python flowScripts/test.py

# 可视化（默认 16 个节点）
python flowScripts/visualize.py
```

## 3. 内生递归多步评估（Flow Evaluation）

该部分用于验证“内生节点递归预测能力”。

协议：

1. 以真实历史窗口作为输入
2. 在一个 block 内递归预测 `steps` 步
3. 外生节点每步使用真值强制输入
4. 内生节点使用模型递归预测值
5. block 结束后重置为真实数据再开始下一个 block

命令：

```bash
# 输出每步误差（CSV + 曲线图）
python flowEvaluation/multi_step_evaluate.py --steps 6

# 输出内生节点递归可视化
python flowEvaluation/visualize_endogenous.py --steps 6 --max-nodes 16
```

输出：

- `flowEvaluation/results/recursive_endo_metrics.csv`
- `flowEvaluation/results/recursive_endo_metrics.png`
- `flowEvaluation/results/endogenous_recursive_visualization.png`

## 4. 充电预测（Charge）

### 4.1 数据准备流程

#### 步骤 A：站点对齐

将 `data/charge/charge.csv` 中不在 canonical 节点集合内的 `node_id` 过滤掉（原地覆盖，可选备份）：

```bash
python chargePrediction/prepare_charge_data.py
```

#### 步骤 B：构建充电训练/测试数据

从流量数据（优先模型预测流量）构建最终训练文件：

```bash
python chargePrediction/prepare_charge_dataset.py
```

默认输出：

- `data/charge/train.csv`
- `data/charge/test.csv`

最终字段：

- `hour_code`
- `week_code`
- `station_id`
- `node_id`
- `charge_flow_ratio`（充电人数 / 交通流）
- `charge_power`（平均功率 = 总功率 / 充电人数）

### 4.2 站点级模型训练

每个站点（`node_id`）训练两个模型（仅使用训练数据，不再拆分验证集）：

1. 比率模型：Beta 分布，学习 `alpha`、`beta`
2. 平均功率模型：LogNormal 分布，学习 `mu`、`sigma`

特征为周/日编码的一倍与二倍频傅里叶基。

```bash
python chargePrediction/train_station_models.py
```

输出：

- `chargePrediction/models/ratio/<node_id>.pt`
- `chargePrediction/models/power/<node_id>.pt`
- `chargePrediction/models/training_summary.csv`
- `chargePrediction/models/training_report.json`

### 4.3 端到端总功率测试（最终指标）

测试集定义：按时间排序后取最后 20%（8:2 切分中的后 2）。

端到端链路：

1. 先用流量模型预测 `pred_gate_flow`
2. 用 Beta 模型预测 `pred_ratio`
3. 用 LogNormal 模型预测 `pred_avg_power`
4. 组合得到总功率预测：
   - `pred_total_power = pred_gate_flow * pred_ratio * pred_avg_power`

评估目标是**总功率**（不再除以充电人数），输出 MAE / RMSE。

```bash
python chargePrediction/test_station_models.py
```

输出：

- `chargePrediction/models/e2e_test_detail.csv`
- `chargePrediction/models/e2e_test_node_metrics.csv`
- `chargePrediction/models/e2e_test_report.json`

### 4.4 端到端可视化

```bash
python chargePrediction/visualize_total_power.py
```

输出：

- `chargePrediction/models/e2e_total_power_visualization.png`

## 5. 主要配置

配置文件：`flowConfig/config.py`

重点配置项：

- 数据与设备：`DATA_DIR`, `MODEL_DIR`, `SEQ_LEN`, `PRED_LEN`, `DEVICE`
- 节点对齐：`NODE_ALIGNMENT`
- Stage-1 外生训练：`EXOGENOUS_TRAIN`
- Stage-2 内生训练：`ENDOGENOUS_TRAIN`
- 混合采样：`SCHEDULED_SAMPLING`
- 非负约束：`OUTPUT_CONSTRAINTS`
- 内生高斯补偿：`ENDOGENOUS_GAUSSIAN`

## 6. 环境依赖

建议 Python 3.10+，核心依赖：

- `torch`
- `numpy`
- `pandas`
- `matplotlib`

可按需安装：

```bash
pip install torch numpy pandas matplotlib
```

## 7. 一键复现实验顺序（建议）

```bash
# 1) 训练交通流模型
python flowScripts/train.py

# 2) 测试与可视化交通流
python flowScripts/test.py
python flowScripts/visualize.py
python flowEvaluation/multi_step_evaluate.py --steps 6
python flowEvaluation/visualize_endogenous.py --steps 6

# 3) 充电数据准备
python chargePrediction/prepare_charge_data.py
python chargePrediction/prepare_charge_dataset.py

# 4) 训练充电模型
python chargePrediction/train_station_models.py

# 5) 端到端总功率测试与可视化
python chargePrediction/test_station_models.py
python chargePrediction/visualize_total_power.py
```

## 8. 备注

- 示例数据仅用于代码连通性验证，真实数据需在服务器环境运行。
- 若本地缺少真实 checkpoint 或完整数据，脚本可能无法得到有效指标，但不影响代码结构与流程正确性。
