import pandas as pd
import os
import numpy as np  # 添加numpy用于subimg reshape
from linear_regression import compute_linear_regression
from random_forest_shap import compute_random_forest_shap
from xgboost_shap import compute_xgboost_shap
from pearson_correlation import compute_pearson_correlation
from plot import plot_feature_counts
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

results_all = {}  # {method_name: (results, feature_counts)}
for compute_func, method_name, color, edgecolor, subdir in methods_config:
    print(f"\n=== 计算 {method_name} ===")
    results, feature_counts = compute_func(df)
    results_all[method_name] = (results, feature_counts)

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

    # 保存feature_stats.csv
    stats_df = pd.DataFrame({'属性': list(feature_counts.keys()), '出现次数': list(feature_counts.values())})
    stats_df.to_csv(os.path.join(subdir_path, 'feature_stats.csv'), index=False)

    # 生成并保存单个图
    fig, ax = plot_feature_counts(feature_counts, method_name, color, edgecolor)
    fig.savefig(os.path.join(subdir_path, 'feature_appearance_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)  # 关闭fig，避免内存泄漏
    print(f"{method_name} 结果已保存到 result/{subdir}/")

# 生成对比图（四个图合成一张）
comparison_dir = 'result/comparison'
os.makedirs(comparison_dir, exist_ok=True)

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

for idx, (method_name, (_, feature_counts)) in enumerate(results_all.items()):
    row = idx // 2
    col = idx % 2
    color, edgecolor, _ = next((c, e, _) for _, n, c, e, _ in methods_config if n == method_name)

    # 生成子图
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

print("\n=== 所有方法运行完成 ===")