import argparse
import os
import numpy as np
import torch
from config import load_config
from data_loader import load_and_preprocess_data
from model import GNN
from train import train_model, evaluate_model
from visualize import (
    plot_loss_curve, plot_pred_vs_true, plot_risk_heatmap,
    plot_residual_histogram, plot_residual_vs_true,
    plot_pred_vs_true_map, plot_residual_heatmap
)
from utils import create_exp_dir, save_json, save_numpy

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'predict'], default='train')
parser.add_argument('--exp_name', type=str, default='', help='实验名称（可选，便于区分）')
args = parser.parse_args()

if __name__ == "__main__":
    # 1. 加载配置
    config = load_config('config.yaml')

    # 2. 创建实验目录（仅训练模式需要）
    if args.mode == 'train':
        base_dir = config.get("save_base_dir", "results")
        exp_dir = create_exp_dir(base_dir)
        if args.exp_name:
            exp_dir = exp_dir.replace("exp_", f"exp_{args.exp_name}_")
        print(f"✅ 所有训练结果将保存至：{exp_dir}")

    # 3. 加载数据
    train_data, train_polygons = load_and_preprocess_data(config['data_path'],exp_dir)
    val_data, val_polygons = load_and_preprocess_data(config['val_data_path'],exp_dir)

    # 4. 初始化模型
    model = GNN(config['in_channels'], config['hidden_channels'])

    # 5. 训练模式
    if args.mode == 'train':
        # 5.1 训练并保存模型
        history = train_model(exp_dir,
            model, train_data, val_data, config,
            save_dir=os.path.join(exp_dir, "model")
        )

        # 5.2 用 best_model 做最终评估
        best_model_path = os.path.join(exp_dir, "model", "best_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        val_pred = evaluate_model(model, val_data)
        val_true = val_data.y.numpy()

        # 5.3 保存数值数据
        save_numpy(np.array(history["train_loss"]),
                   os.path.join(exp_dir, "data", "train_losses.npy"))
        save_numpy(np.array(history["val_loss"]),
                   os.path.join(exp_dir, "data", "val_losses.npy"))
        metrics = {
            "val_MSE": float(np.mean((val_true - val_pred) ** 2)),
            "val_MAE": float(np.mean(np.abs(val_true - val_pred))),
            "val_R²": float(1 - np.sum((val_true - val_pred) ** 2) / np.sum((val_true - np.mean(val_true)) ** 2)),
            "epochs": config["epochs"],
            "learning_rate": config["learning_rate"]
        }
        save_json(metrics, os.path.join(exp_dir, "data", "val_metrics.json"))
        print(f"📊 数值数据已保存至：{os.path.join(exp_dir, 'data')}")

        # 5.4 保存可视化图表
        # (a) 损失收敛曲线（全程）
        plot_loss_curve(
            history["train_loss"], history["val_loss"],
            save_path=os.path.join(exp_dir, "plots", "loss_curve_all.png")
        )
        # (b) 前200轮的损失曲线
        plot_loss_curve(
            history["train_loss"][:200], history["val_loss"][:200],
            save_path=os.path.join(exp_dir, "plots", "loss_curve_200.png")
        )
        # (c) 预测 vs 真实
        plot_pred_vs_true(val_true, val_pred,
                          save_path=os.path.join(exp_dir, "plots", "val_pred_vs_true.png"))
        # (d) 风险热力图
        plot_risk_heatmap(val_polygons, val_pred,
                          save_path=os.path.join(exp_dir, "plots", "val_risk_heatmap.png"))
        # (e) 残差直方图
        plot_residual_histogram(val_true, val_pred,
                                save_path=os.path.join(exp_dir, "plots", "residual_histogram.png"))
        # (f) 残差 vs 真实值
        plot_residual_vs_true(val_true, val_pred,
                              save_path=os.path.join(exp_dir, "plots", "residual_vs_true.png"))
        # (g) 真实 vs 预测 空间分布
        plot_pred_vs_true_map(val_polygons, val_true, val_pred,
                              save_path=os.path.join(exp_dir, "plots", "pred_vs_true_map.png"))
        # (h) 残差空间分布
        plot_residual_heatmap(val_polygons, val_true, val_pred,
                              save_path=os.path.join(exp_dir, "plots", "residual_heatmap.png"))

