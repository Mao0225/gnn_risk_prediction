import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import matplotlib
from matplotlib.colors import ListedColormap

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


import geopandas as gpd  # 导入geopandas库，用于处理地理数据
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np  # 导入numpy库，用于数组操作（如计算绝对值和最大值）
from matplotlib.colors import LinearSegmentedColormap  # 导入LinearSegmentedColormap，用于自定义连续颜色映射


def plot_residual_heatmap(polygons, y_true, y_pred, save_path=None):
    """
    绘制残差（预测值-真实值）的空间分布热力图，分析误差的地理分布特征

    参数:
        polygons: 多边形几何对象列表，代表地理区域（例如shapely Polygon对象）
        y_true: 真实值数组（numpy数组或列表，与polygons长度一致）
        y_pred: 预测值数组（numpy数组或列表，与polygons长度一致）
        save_path: 图像保存路径，若为None则直接显示图像（str或None）

    返回:
        无（直接显示或保存图像）

    示例:
        plot_residual_heatmap(polygons_list, true_values, pred_values, 'residual_heatmap.png')
    """
    # 计算残差：预测值减去真实值，得到误差数组
    residuals = np.array(y_pred) - np.array(y_true)  # 转换为numpy数组，确保兼容

    # 创建地理数据框（GeoDataFrame），将几何对象和残差值关联起来
    # geometry列存储多边形，residual列存储对应残差值
    gdf = gpd.GeoDataFrame({"geometry": polygons, "residual": residuals})

    # 强制对称颜色范围：vmin = -max(|residuals|), vmax = max(|residuals|)，确保0落在中心（白色）
    abs_res = np.abs(residuals)
    vmax = np.max(abs_res)
    vmin = -vmax

    # 自定义蓝-白-红colormap：严格定义中心为纯白（通过高色阶数减少偏差）
    # cdict定义：0.0=蓝(0,0,1)，0.5=白(1,1,1)，1.0=红(1,0,0)
    cdict = {
        'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0, 0.0, 0.0)]
    }
    cmap = LinearSegmentedColormap('custom_bwr', cdict, N=1000)  # N=1000增加色阶，使中心更精确为白

    # 绘制热力图：使用residual列着色，指定对称vmin/vmax和自定义cmap，legend显示图例，figsize设置图像大小
    ax = gdf.plot(column="residual", cmap=cmap, vmin=vmin, vmax=vmax, legend=True, figsize=(8, 6))
    ax.set_title("残差空间分布")  # 设置图像标题

    # 如果指定保存路径，则保存图像（高分辨率，无边框），否则直接显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图像，避免内存泄漏
    else:
        plt.show()  # 显示图像