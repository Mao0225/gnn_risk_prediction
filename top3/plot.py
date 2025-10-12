import matplotlib.pyplot as plt

# 修复中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_feature_counts(feature_counts, method_name, color='skyblue', edgecolor='navy'):
    """
    生成单个方法的条形图。
    :param feature_counts: dict {feat: count}, 已降序
    :param method_name: str, 方法名
    :param color: str, 条形颜色
    :param edgecolor: str, 边框颜色
    :return: fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    features = list(feature_counts.keys())
    counts = list(feature_counts.values())
    bars = ax.bar(features, counts, color=color, edgecolor=edgecolor)
    ax.set_xlabel('属性')
    ax.set_ylabel('在前三中的出现次数')
    ax.set_title(f'每个属性在前三贡献度中的出现次数统计 ({method_name})')
    ax.tick_params(axis='x', rotation=45, labelrotation=45)  # 移除无效的ha，保留rotation和labelrotation
    # 修复ha：单独设置xticklabels的对齐
    for label in ax.get_xticklabels():
        label.set_ha('right')
    plt.tight_layout()
    return fig, ax