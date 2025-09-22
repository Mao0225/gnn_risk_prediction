import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import os

# ===== 解决中文显示问题 =====
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置高DPI显示
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300


def visualize_risk(file_path, save_dir):
    """
    读取风险数据并绘制可视化图表（针对渔网数据优化）
    file_path: 输入数据文件（csv 或 excel）
    save_dir: 图像保存路径
    """
    # 自动判断文件类型
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # 去掉列名前后空格
    df.rename(columns=lambda x: str(x).strip(), inplace=True)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    print("当前表格列名:", df.columns.tolist())

    # === 计算误差 ===
    if "差值" not in df.columns:
        df["差值"] = df["预测风险值"] - df["风险值"]
    if "绝对差值" not in df.columns:
        df["绝对差值"] = df["差值"].abs()

    # === 1. 散点图（风险值 vs 预测风险值）===
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=df["风险值"], y=df["预测风险值"],
        alpha=0.8, s=120, color="royalblue", edgecolor="darkblue", linewidth=1
    )
    plt.plot([df["风险值"].min(), df["风险值"].max()],
             [df["风险值"].min(), df["风险值"].max()], "r--", lw=3, alpha=0.8)
    plt.xlabel("真实风险值", fontsize=12, weight="bold")
    plt.ylabel("预测风险值", fontsize=12, weight="bold")
    plt.title("真实 vs 预测风险值对比", fontsize=16, weight="bold", pad=20)
    plt.grid(alpha=0.3, linewidth=1)

    # 添加R²和RMSE信息
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(df["风险值"], df["预测风险值"])
    rmse = np.sqrt(mean_squared_error(df["风险值"], df["预测风险值"]))
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "散点图_真实_vs_预测.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === 2. 误差直方图 ===
    plt.figure(figsize=(8, 5))
    sns.histplot(df["差值"], bins=30, kde=True,
                 color="orange", edgecolor="black", alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.xlabel("差值 (预测 - 真实)", fontsize=12, weight="bold")
    plt.ylabel("频次", fontsize=12, weight="bold")
    plt.title("预测误差分布", fontsize=16, weight="bold", pad=20)
    plt.grid(alpha=0.3)

    # 添加统计信息
    mean_error = df["差值"].mean()
    std_error = df["差值"].std()
    plt.text(0.05, 0.95, f'均值: {mean_error:.3f}\n标准差: {std_error:.3f}',
             transform=plt.gca().transAxes, fontsize=11,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "误差直方图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === 3. 自动检测坐标列 ===
    coord_col = None
    for col in df.columns:
        if "object" in col.lower() or "坐标" in col:
            coord_col = col
            break

    if coord_col is None:
        raise KeyError("❌ 没有找到坐标列（需要包含 'object' 或 '坐标'）")

    print(f"使用的坐标列: {coord_col}")

    # 转换成 Polygon
    def parse_coords(x):
        if isinstance(x, str):
            coords = eval(x)
        else:
            coords = x
        return Polygon(coords[0])

    df["geometry"] = df[coord_col].apply(parse_coords)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    # === 4. 优化的空间可视化 ===

    # --- 真实风险 vs 预测风险（并排对比图） ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # 真实风险图
    im1 = gdf.plot(column="风险值", cmap="plasma", legend=False,
                   ax=axes[0], linewidth=0, alpha=0.9)
    axes[0].set_title("真实风险分布", fontsize=18, weight="bold", pad=20)
    axes[0].axis('off')  # 去掉坐标轴

    # 添加颜色条
    sm1 = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=gdf["风险值"].min(),
                                                   vmax=gdf["风险值"].max()))
    sm1._A = []
    cbar1 = fig.colorbar(sm1, ax=axes[0], shrink=0.8, aspect=30)
    cbar1.set_label('风险值', fontsize=14, weight="bold")
    cbar1.ax.tick_params(labelsize=12)

    # 预测风险图
    im2 = gdf.plot(column="预测风险值", cmap="plasma", legend=False,
                   ax=axes[1], linewidth=0, alpha=0.9)
    axes[1].set_title("预测风险分布", fontsize=18, weight="bold", pad=20)
    axes[1].axis('off')  # 去掉坐标轴

    # 添加颜色条
    sm2 = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=gdf["预测风险值"].min(),
                                                   vmax=gdf["预测风险值"].max()))
    sm2._A = []
    cbar2 = fig.colorbar(sm2, ax=axes[1], shrink=0.8, aspect=30)
    cbar2.set_label('预测风险值', fontsize=14, weight="bold")
    cbar2.ax.tick_params(labelsize=12)

    plt.suptitle("真实 vs 预测 风险空间分布对比", fontsize=20, weight="bold", y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "地图_真实_vs_预测.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 绝对误差空间分布 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 使用从白色到深红色的颜色映射
    gdf.plot(
        column="绝对差值",
        cmap="Reds", legend=False,
        vmin=0, vmax=df["绝对差值"].max(),
        ax=ax, linewidth=0, alpha=0.9
    )
    ax.set_title("绝对误差空间分布", fontsize=18, weight="bold", pad=20)
    ax.axis('off')  # 去掉坐标轴

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap="Reds",
                               norm=plt.Normalize(vmin=0, vmax=df["绝对差值"].max()))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('绝对误差', fontsize=14, weight="bold")
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "地图_绝对误差.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === 5. 新增：差值空间分布（蓝-白-红）===
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 计算差值的最大绝对值，用于对称的颜色映射
    max_abs_diff = max(abs(df["差值"].min()), abs(df["差值"].max()))

    gdf.plot(
        column="差值",
        cmap="RdBu_r",  # 红-白-蓝颜色映射，红色表示过估计，蓝色表示低估计
        legend=False,
        vmin=-max_abs_diff, vmax=max_abs_diff,  # 对称范围
        ax=ax, linewidth=0, alpha=0.9
    )
    ax.set_title("预测偏差空间分布\n(红色=过估计，蓝色=低估计)", fontsize=18, weight="bold", pad=20)
    ax.axis('off')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                               norm=plt.Normalize(vmin=-max_abs_diff, vmax=max_abs_diff))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('差值 (预测-真实)', fontsize=14, weight="bold")
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "地图_预测偏差.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === 6. 新增：高误差区域识别 ===
    # 标记误差大于75%分位数的区域
    high_error_threshold = df["绝对差值"].quantile(0.75)
    df["高误差区域"] = df["绝对差值"] > high_error_threshold
    gdf["高误差区域"] = df["高误差区域"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 先画低误差区域（灰色）
    gdf[~gdf["高误差区域"]].plot(ax=ax, color='lightgray', linewidth=0, alpha=0.5)

    # 再画高误差区域（红色）
    gdf[gdf["高误差区域"]].plot(ax=ax, color='red', linewidth=0, alpha=0.8)

    ax.set_title(f"高误差区域识别\n(红色区域误差 > {high_error_threshold:.3f})",
                 fontsize=18, weight="bold", pad=20)
    ax.axis('off')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.8, label=f'高误差区域 (n={sum(df["高误差区域"])})'),
                       Patch(facecolor='lightgray', alpha=0.5, label=f'正常区域 (n={sum(~df["高误差区域"])})')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "地图_高误差区域.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === 7. 输出统计摘要 ===
    print("\n" + "=" * 50)
    print("📊 统计摘要")
    print("=" * 50)
    print(f"总网格数量: {len(df)}")
    print(f"R² 决定系数: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"平均绝对误差: {df['绝对差值'].mean():.4f}")
    print(f"最大绝对误差: {df['绝对差值'].max():.4f}")
    print(f"高误差区域数量: {sum(df['高误差区域'])} ({sum(df['高误差区域']) / len(df) * 100:.1f}%)")
    print(f"误差标准差: {df['差值'].std():.4f}")

    print(f"\n✅ 所有图表已保存到: {save_dir}")
    print("📁 生成的图表文件:")
    print("  • 散点图_真实_vs_预测.png")
    print("  • 误差直方图.png")
    print("  • 地图_真实_vs_预测.png")
    print("  • 地图_绝对误差.png")
    print("  • 地图_预测偏差.png")
    print("  • 地图_高误差区域.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python predict_plot.py <结果文件.xlsx> <输出目录>")
        sys.exit(1)
    file_path = sys.argv[1]
    save_dir = sys.argv[2]
    visualize_risk(file_path, save_dir)