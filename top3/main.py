import pandas as pd
import os
import numpy as np  # 添加numpy用于subimg reshape
from linear_regression import compute_linear_regression
from random_forest_shap import compute_random_forest_shap
from xgboost_shap import compute_xgboost_shap
from pearson_correlation import compute_pearson_correlation
from plot import plot_feature_counts, plot_avg_contrib_rates  # 添加新plot函数
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载（抽到main.py）
file_path = '../data/test.xlsx'
df = pd.read_excel(file_path)
print(f"数据行数: {len(df)}")
print(df.head())

# 创建result文件夹
os.makedirs('result', exist_ok=True)

# 方法配置：(函数, 方法名, 颜色, 边框色, 子文件夹)
methods_config = [
    (compute_linear_regression, '线性回归', 'skyblue', 'navy', 'functionA'),
    (compute_random_forest_shap, '随机森林 + SHAP', 'lightgreen', 'darkgreen', 'functionB'),
    (compute_xgboost_shap, 'XGBoost + SHAP', 'lightcoral', 'darkred', 'functionC'),
    (compute_pearson_correlation, 'Pearson相关分析', 'lightblue', 'darkblue', 'functionD')
]

results_all = {}  # {method_name: (results, feature_counts, avg_contrib_rates)}
for compute_func, method_name, color, edgecolor, subdir in methods_config:
    print(f"\n=== 计算 {method_name} ===")
    results, feature_counts, avg_contrib_rates = compute_func(df)  # 统一解包3个值
    results_all[method_name] = (results, feature_counts, avg_contrib_rates)

    # 创建子文件夹并保存CSV
    subdir_path = os.path.join('result', subdir)
    os.makedirs(subdir_path, exist_ok=True)

    # 保存top_influences.csv
    results_df = pd.DataFrame([
        {'grid_id': r['grid_id'], 'risk_value': r['risk_value'],
         **{f'top_feat_{i + 1}': feat for i, (feat, _) in enumerate(r['top_3_influences'])},
         **{f'top_contrib_{i + 1}': contrib for i, (_, contrib) in enumerate(r['top_3_influences'])}}
        for r in results
    ])
    results_df.to_csv(os.path.join(subdir_path, 'top_influences.csv'), index=False)

    # 保存feature_stats.csv (出现次数)
    stats_df = pd.DataFrame({'属性': list(feature_counts.keys()), '出现次数': list(feature_counts.values())})
    stats_df.to_csv(os.path.join(subdir_path, 'feature_stats.csv'), index=False)

    # 保存avg_contrib_rates.csv
    contrib_df = pd.DataFrame({'属性': list(avg_contrib_rates.keys()), '平均贡献率(%)': list(avg_contrib_rates.values())})
    contrib_df.to_csv(os.path.join(subdir_path, 'avg_contrib_rates.csv'), index=False)

    # 生成并保存出现次数图
    fig, ax = plot_feature_counts(feature_counts, method_name, color, edgecolor)
    fig.savefig(os.path.join(subdir_path, 'feature_appearance_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 生成并保存平均贡献率图
    fig2, ax2 = plot_avg_contrib_rates(avg_contrib_rates, method_name, color, edgecolor)
    fig2.savefig(os.path.join(subdir_path, 'avg_contrib_rates_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"{method_name} 结果已保存到 result/{subdir}/")

# 生成对比图（四个出现次数图合成一张）
comparison_dir = 'result/comparison'
os.makedirs(comparison_dir, exist_ok=True)

# 出现次数对比图
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

for idx, (method_name, (results, feature_counts, _)) in enumerate(results_all.items()):
    row = idx // 2
    col = idx % 2
    color, edgecolor, _ = next((c, e, _) for _, n, c, e, _ in methods_config if n == method_name)

    # 生成子图 (出现次数)
    subfig, subax = plot_feature_counts(feature_counts, method_name, color, edgecolor)
    subfig.canvas.draw()  # 渲染子图

    # 修复：转换为NumPy数组后切片
    buf = np.asarray(subfig.canvas.renderer.buffer_rgba())  # (H, W, 4)
    height, width = buf.shape[:2]
    subimg = buf[:, :, :3]  # 提取 RGB 通道 (H, W, 3)

    # 添加到主图的子plot
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(subimg)
    ax.set_title(method_name, fontsize=14, pad=20)
    ax.axis('off')  # 隐藏轴
    plt.close(subfig)  # 关闭子fig

plt.suptitle('四个方法属性出现次数对比', fontsize=16, y=0.98)
plt.tight_layout()
comparison_path = os.path.join(comparison_dir, 'comparison_bar_charts.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"对比图已保存到 {comparison_path}")

# 平均贡献率对比图（类似结构）
fig2 = plt.figure(figsize=(20, 12))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.2)

for idx, (method_name, (_, _, avg_contrib_rates)) in enumerate(results_all.items()):
    row = idx // 2
    col = idx % 2
    color, edgecolor, _ = next((c, e, _) for _, n, c, e, _ in methods_config if n == method_name)

    # 生成子图 (平均贡献率)
    subfig2, subax2 = plot_avg_contrib_rates(avg_contrib_rates, method_name, color, edgecolor)
    subfig2.canvas.draw()  # 渲染子图

    # 修复：转换为NumPy数组后切片
    buf2 = np.asarray(subfig2.canvas.renderer.buffer_rgba())  # (H, W, 4)
    height2, width2 = buf2.shape[:2]
    subimg2 = buf2[:, :, :3]  # 提取 RGB 通道 (H, W, 3)

    # 添加到主图的子plot
    ax2 = fig2.add_subplot(gs2[row, col])
    ax2.imshow(subimg2)
    ax2.set_title(method_name, fontsize=14, pad=20)
    ax2.axis('off')  # 隐藏轴
    plt.close(subfig2)  # 关闭子fig

plt.suptitle('四个方法平均贡献率对比', fontsize=16, y=0.98)
plt.tight_layout()
comparison_contrib_path = os.path.join(comparison_dir, 'comparison_avg_contrib_rates_bar_charts.png')
plt.savefig(comparison_contrib_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"平均贡献率对比图已保存到 {comparison_contrib_path}")

print("\n=== 所有方法运行完成 ===")