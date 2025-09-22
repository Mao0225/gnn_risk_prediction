import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import matplotlib

# 中文显示设置：确保图表中中文正常显示，避免乱码
# 设置默认字体为SimHei（黑体）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_loss_curve(train_losses, val_losses, save_path=None, first_n=200):
    """
    绘制训练和验证损失曲线，包括整体曲线和前N轮细节曲线

    参数:
        train_losses: 训练损失列表，每个元素对应一轮的训练损失
        val_losses: 验证损失列表，每个元素对应一轮的验证损失
        save_path: 图像保存路径，若为None则直接显示图像
        first_n: 前N轮损失曲线的显示轮次，默认200轮
    """
    # 生成epochs数组（从1开始的连续整数）
    epochs = range(1, len(train_losses) + 1)

    # 绘制整体损失曲线
    plt.figure(figsize=(8, 6))  # 设置图像大小
    plt.plot(epochs, train_losses, label='训练损失')  # 绘制训练损失
    plt.plot(epochs, val_losses, label='验证损失')  # 绘制验证损失
    plt.xlabel("Epoch")  # x轴标签
    plt.ylabel("Loss")  # y轴标签
    plt.title("训练 & 验证损失曲线（整体）")  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线

    # 根据save_path决定保存还是显示图像
    if save_path:
        # 保存整体曲线，文件名添加"_full"标识
        plt.savefig(save_path.replace(".png", "_full.png"), dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图像释放内存
    else:
        plt.show()  # 直接显示图像

    # 绘制前N轮损失曲线（仅当训练轮次超过first_n时）
    if len(train_losses) > first_n:
        plt.figure(figsize=(8, 6))
        # 只取前first_n个数据点
        plt.plot(epochs[:first_n], train_losses[:first_n], label='训练损失')
        plt.plot(epochs[:first_n], val_losses[:first_n], label='验证损失')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"训练 & 验证损失曲线（前 {first_n} 轮）")
        plt.legend()
        plt.grid(True)

        if save_path:
            # 保存前N轮曲线，文件名添加"_firstN"标识
            plt.savefig(save_path.replace(".png", f"_first{first_n}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_pred_vs_true(y_true, y_pred, save_path=None):
    """
    绘制预测值与真实值的散点图，用于直观评估预测效果

    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        save_path: 图像保存路径，若为None则直接显示图像
    """
    plt.figure(figsize=(6, 6))  # 正方形画布，便于观察偏离程度
    # 绘制散点图，alpha设置透明度避免点重叠
    plt.scatter(y_true, y_pred, alpha=0.5)
    # 绘制对角线（y=x），理想情况下所有点应分布在这条线上
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             'r--', lw=2)  # 红色虚线，线宽2
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.title("预测 vs 真实")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residual_histogram(y_true, y_pred, save_path=None):
    """
    绘制残差（预测值-真实值）的直方图，分析误差分布特征

    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        save_path: 图像保存路径，若为None则直接显示图像
    """
    # 计算残差：预测值减去真实值
    residuals = y_pred - y_true

    plt.figure(figsize=(8, 6))
    # 绘制直方图，设置50个 bins 平衡细节和可读性
    plt.hist(residuals, bins=50, color="steelblue", alpha=0.7)
    # 添加参考线：y=0处的红色虚线
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("残差 (预测 - 真实)")
    plt.ylabel("频率")
    plt.title("残差分布直方图")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residual_vs_true(y_true, y_pred, save_path=None):
    """
    绘制残差与真实值的散点图，分析误差是否随真实值变化而呈现规律

    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        save_path: 图像保存路径，若为None则直接显示图像
    """
    # 计算残差：预测值减去真实值
    residuals = y_pred - y_true

    plt.figure(figsize=(8, 6))
    # 绘制散点图，alpha设置透明度
    plt.scatter(y_true, residuals, alpha=0.5)
    # 添加参考线：y=0处的红色虚线
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("真实值")
    plt.ylabel("残差 (预测 - 真实)")
    plt.title("残差 vs 真实值")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(history, save_path=None):
    """
    绘制训练过程中的评估指标曲线（MSE、MAE、R²）

    参数:
        history: 字典，包含训练过程中记录的指标，需包含以下键：
                 - "epoch": 轮次列表
                 - "val_MSE": 验证集MSE列表
                 - "val_MAE": 验证集MAE列表
                 - "val_R2": 验证集R²列表
        save_path: 图像保存路径，若为None则直接显示图像
    """
    epochs = history["epoch"]  # 获取轮次数据

    plt.figure(figsize=(14, 4))  # 宽屏画布，容纳3个子图

    # 绘制MSE曲线（子图1）
    plt.subplot(1, 3, 1)  # 1行3列中的第1个
    plt.plot(epochs, history["val_MSE"], label="验证MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE 曲线")
    plt.grid(True)

    # 绘制MAE曲线（子图2）
    plt.subplot(1, 3, 2)  # 1行3列中的第2个
    plt.plot(epochs, history["val_MAE"], label="验证MAE", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("MAE 曲线")
    plt.grid(True)

    # 绘制R²曲线（子图3）
    plt.subplot(1, 3, 3)  # 1行3列中的第3个
    plt.plot(epochs, history["val_R2"], label="验证R²", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("R² 曲线")
    plt.grid(True)

    plt.tight_layout()  # 自动调整子图布局，避免重叠

    if save_path:
        plt.savefig(save_path.replace(".png", "_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_risk_heatmap(polygons, values, save_path=None):
    """
    绘制风险值的空间热力图，展示风险的地理分布特征

    参数:
        polygons: 多边形几何对象列表，代表地理区域
        values: 与多边形对应的风险值数组
        save_path: 图像保存路径，若为None则直接显示图像
    """
    # 创建地理数据框（GeoDataFrame），关联几何对象和风险值
    gdf = gpd.GeoDataFrame({"geometry": polygons, "risk": values})
    # 绘制热力图，使用Reds配色方案（红色越深风险越高）
    ax = gdf.plot(column="risk", cmap="Reds", legend=True, figsize=(8, 6))
    ax.set_title("风险热力图")  # 设置标题

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pred_vs_true_map(polygons, y_true, y_pred, save_path=None):
    """
    对比绘制真实风险值和预测风险值的空间分布地图

    参数:
        polygons: 多边形几何对象列表，代表地理区域
        y_true: 真实风险值数组
        y_pred: 预测风险值数组
        save_path: 图像保存路径，若为None则直接显示图像
    """
    # 创建两个地理数据框，分别存储真实值和预测值
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1行2列的子图布局
    gdf_true = gpd.GeoDataFrame({"geometry": polygons, "risk": y_true})
    gdf_pred = gpd.GeoDataFrame({"geometry": polygons, "risk": y_pred})

    # 绘制真实风险分布图（左图）
    gdf_true.plot(column="risk", cmap="Blues", legend=True, ax=axes[0])
    axes[0].set_title("真实风险分布")

    # 绘制预测风险分布图（右图）
    gdf_pred.plot(column="risk", cmap="Blues", legend=True, ax=axes[1])
    axes[1].set_title("预测风险分布")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residual_heatmap(polygons, y_true, y_pred, save_path=None):
    """
    绘制残差（预测值-真实值）的空间分布热力图，分析误差的地理分布特征

    参数:
        polygons: 多边形几何对象列表，代表地理区域
        y_true: 真实值数组
        y_pred: 预测值数组
        save_path: 图像保存路径，若为None则直接显示图像
    """
    # 计算残差：预测值减去真实值
    residuals = y_pred - y_true

    # 创建地理数据框，关联几何对象和残差值
    gdf = gpd.GeoDataFrame({"geometry": polygons, "residual": residuals})
    # 使用bwr配色方案（蓝-白-红），便于区分正负残差
    ax = gdf.plot(column="residual", cmap="bwr", legend=True, figsize=(8, 6))
    ax.set_title("残差空间分布")  # 设置标题

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()