"""
递归预测测试脚本
- 外生节点：使用LSTM预测（每个外生节点有独立的LSTM）
- 内生节点：通过递归方式预测4步（测试模型递归能力）
- 可视化结果
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import flowConfig.config as config
from src.data.flow_data import FlowDataLoader, create_sequences
from src.models.flow_predictor import FlowPredictor


def prepare_test_data(loader: FlowDataLoader, topo_order: list, norm_stats: dict = None):
    """准备测试数据 - 填充到完整节点数"""
    test_data = loader.load_flow_data("test.csv")

    numeric_cols = test_data.select_dtypes(include=[np.number]).columns
    test_cols = [col for col in numeric_cols if col != 'timestamp']
    test_cols_set = set(test_cols)

    num_total = len(topo_order)
    print(f"Train nodes: {num_total}, Test nodes: {len(test_cols)}")

    # 填充到完整维度
    aligned_data = np.zeros((len(test_data), num_total))
    for i, col in enumerate(topo_order):
        if col in test_cols_set:
            aligned_data[:, i] = test_data[col].values

    mean = norm_stats['mean']
    std = norm_stats['std']
    normalized_data = (aligned_data - mean) / std

    SEQ_LEN = config.SEQ_LEN
    PRED_LEN = config.PRED_LEN

    X, Y = create_sequences(normalized_data, SEQ_LEN, PRED_LEN)
    hours = np.arange(len(X)) % 24

    valid_mask = np.array([col in test_cols_set for col in topo_order])

    return X, Y, hours, mean, std, valid_mask, topo_order


def recursive_predict(model, x_hist, hour, device, num_exogenous, num_recursive_steps=4):
    """递归预测"""
    model.eval()
    all_predictions = []
    current_x = x_hist.clone()

    for step in range(num_recursive_steps):
        with torch.no_grad():
            pred = model(current_x, hour)
            pred = pred.squeeze(1)
            all_predictions.append(pred)

            current_x = torch.cat([current_x[:, 1:, :], pred.unsqueeze(1)], dim=1)
            hour = hour + 1

    predictions = torch.stack(all_predictions, dim=1)
    return predictions


def main():
    device = config.get_device()
    print(f"Using device: {device}")

    loader = FlowDataLoader(config.DATA_DIR)
    model_path = f"{config.MODEL_DIR}/best_model.pt"

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        return

    topo_order = checkpoint['topo_order']
    exogenous_nodes = checkpoint['exogenous_nodes']
    upstream_dict = checkpoint.get('upstream_dict', {})
    norm_stats = checkpoint.get('norm_stats')

    print(f"Checkpoint: {len(topo_order)} nodes, {len(exogenous_nodes)} exogenous")

    # 重建模型
    model = FlowPredictor(
        num_nodes=len(topo_order),
        node_ids=topo_order,
        exogenous_nodes=exogenous_nodes,
        topo_order=topo_order,
        upstream_dict=upstream_dict
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("\nLoading test data...")
    X_test, Y_test, hours_test, mean, std, valid_mask, data_cols = prepare_test_data(
        loader, topo_order, norm_stats
    )

    print(f"Test: X={X_test.shape}, Y={Y_test.shape}")

    exogenous_set = set(exogenous_nodes)

    # 随机抽取16个站点
    np.random.seed(42)
    valid_indices = np.where(valid_mask)[0]
    exo_in_valid = [i for i in valid_indices if data_cols[i] in exogenous_set]
    endo_in_valid = [i for i in valid_indices if data_cols[i] not in exogenous_set]

    print(f"Valid exogenous: {len(exo_in_valid)}, Valid endogenous: {len(endo_in_valid)}")

    selected_exo = np.random.choice(exo_in_valid, size=min(4, len(exo_in_valid)), replace=False).tolist() if exo_in_valid else []
    remaining = 16 - len(selected_exo)
    selected_endo = np.random.choice(endo_in_valid, size=remaining, replace=False).tolist() if endo_in_valid else []
    selected_nodes = np.array(selected_exo + selected_endo)

    print(f"Selected nodes: {[data_cols[i] for i in selected_nodes]}")

    # 递归预测
    print("\nRunning recursive prediction (4 steps)...")
    X_tensor = torch.FloatTensor(X_test).to(device)
    hours_tensor = torch.FloatTensor(hours_test[:, None]).to(device)

    recursive_preds = recursive_predict(model, X_tensor, hours_tensor, device, len(exogenous_nodes), 4)
    recursive_preds = recursive_preds.cpu().numpy()

    with torch.no_grad():
        onestep_preds = model(X_tensor, hours_tensor).cpu().numpy()

    # 反标准化
    Y_true = Y_test * std + mean
    Y_recursive = recursive_preds * std + mean
    Y_onestep = onestep_preds * std + mean

    # 计算误差
    print("\n" + "="*50)
    print("Recursive Prediction Results (4 steps)")
    print("="*50)

    exo_selected = [i for i in selected_nodes if data_cols[i] in exogenous_set]
    endo_selected = [i for i in selected_nodes if data_cols[i] not in exogenous_set]

    for step in range(4):
        step_true = Y_true[:, 0, :]
        step_recursive = Y_recursive[:, step, :]
        if endo_selected:
            endo_mae = np.mean(np.abs(step_recursive[:, endo_selected] - step_true[:, endo_selected]))
            print(f"Step {step+1} Endogenous MAE: {endo_mae:.4f}")

    onestep_true = Y_true[:, 0, :]
    if endo_selected:
        onestep_mae = np.mean(np.abs(Y_onestep[:, 0, endo_selected] - onestep_true[:, endo_selected]))
        print(f"One-step Endogenous MAE: {onestep_mae:.4f}")

    # 可视化
    vis_endo = endo_selected[:4] if endo_selected else []
    vis_exo = exo_selected[:4] if exo_selected else []
    vis_nodes = vis_exo + vis_endo

    if len(vis_nodes) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, node_idx in enumerate(vis_nodes[:4]):
            ax = axes[i]
            node_id = data_cols[node_idx]
            is_exogenous = node_id in exogenous_set

            max_points = min(200, Y_true.shape[0])
            time_steps = np.arange(max_points)

            true_vals = Y_true[:max_points, 0, node_idx]
            recursive_vals = [Y_recursive[:max_points, step, node_idx] for step in range(4)]
            onestep_val = Y_onestep[:max_points, 0, node_idx]

            ax.plot(time_steps, true_vals, 'b-', label='True', linewidth=1.5, alpha=0.8)
            ax.plot(time_steps, onestep_val, 'g--', label='One-step Pred', linewidth=1, alpha=0.6)

            colors = ['r', 'orange', 'purple', 'brown']
            for step in range(4):
                ax.plot(time_steps, recursive_vals[step], color=colors[step],
                       linestyle=':', label=f'Recursive Step {step+1}', linewidth=1, alpha=0.7)

            if is_exogenous:
                ax.set_title(f'Node {node_id} [外生-LSTM*]', color='red', fontweight='bold')
            else:
                ax.set_title(f'Node {node_id} [内生-传播]', fontweight='bold')

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Flow')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle('递归预测 vs 一次性预测\n(* 外生节点: LSTM预测, 内生节点: 贝叶斯传播)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{config.MODEL_DIR}/recursive_visualization.png', dpi=150)
        print(f"\nVisualization saved to {config.MODEL_DIR}/recursive_visualization.png")
        plt.close()


if __name__ == "__main__":
    main()
