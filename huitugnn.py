"""
Graph Neural Network Architecture Visualization
Based on Gavin Ding's ConvNet drawing script (modified for GNN)
Author: Carl Ma (with GPT-5 assistance)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# 颜色定义
White = 1.0
Light = 0.7
Medium = 0.5
Dark = 0.3
Black = 0.0

def add_block(patches, colors, xy, size=(20, 20), color=Light, label_text=""):
    """添加矩形模块"""
    rect = Rectangle(xy, size[0], size[1], edgecolor="black", facecolor=str(color))
    patches.append(rect)
    colors.append(color)
    plt.text(
        xy[0] + size[0] / 2,
        xy[1] + size[1] / 2,
        label_text,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        family="sans-serif"
    )

def add_arrow(p1, p2):
    """添加层间箭头"""
    plt.arrow(
        p1[0], p1[1],
        p2[0] - p1[0], p2[1] - p1[1],
        length_includes_head=True,
        head_width=2,
        head_length=3,
        fc="black", ec="black"
    )

if __name__ == "__main__":
    # 初始化图像
    fig, ax = plt.subplots(figsize=(10, 3))
    patches, colors = [], []

    # 坐标配置
    x_start = 0
    y = 0
    width = 22
    height = 18
    gap = 10

    # 各层位置与标签
    layers = [
        ("Input Features\n[N × F_in]", Light),
        ("GraphConv1\n(F_in → Hidden)", Medium),
        ("BatchNorm\n+ ReLU\n+ Dropout", Light),
        ("GraphConv2\n(Hidden → 1)", Medium),
        ("Output\n[N]", Light),
    ]

    # 绘制每一层
    for i, (text, color) in enumerate(layers):
        add_block(patches, colors, xy=(x_start + i * (width + gap), y), size=(width, height), color=color, label_text=text)
        if i > 0:
            add_arrow(
                (x_start + (i - 1) * (width + gap) + width, y + height / 2),
                (x_start + i * (width + gap), y + height / 2)
            )

    # 添加标题
    plt.text(10, height + 10, "GNN Architecture (2-layer GraphConv Network)", fontsize=12, fontweight="bold")

    # 去除坐标轴
    plt.axis("equal")
    plt.axis("off")

    # 保存与显示
    plt.tight_layout()
    save_path = os.path.join("./", "gnn_architecture.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.show()

    print(f"✅ 图像已保存到 {save_path}")
