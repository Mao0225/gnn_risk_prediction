import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import pearsonr

def compute_pearson_correlation(df):
    # 定义特征列
    feature_cols = [col for col in df.columns if col not in ['Id', '风险值','Object','FID','属于']]

    # 准备X和y
    X = df[feature_cols]
    y = df['风险值']

    # 标准化
    use_scaling = True
    X_for_contrib = X
    if use_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_for_contrib = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    # 计算全局相关系数
    corrs = [abs(pearsonr(X[col], y)[0]) for col in feature_cols]
    corrs = np.array(corrs)

    # 计算贡献度
    contributions = np.abs(X_for_contrib.values * corrs)

    # 提取前三
    feature_names = feature_cols
    results = []
    for i in range(len(df)):
        grid_contribs = contributions[i]
        top_indices = np.argsort(grid_contribs)[::-1][:3]
        top_features = [(feature_names[idx], grid_contribs[idx]) for idx in top_indices]
        results.append({
            'grid_id': df.iloc[i]['Id'],
            'risk_value': df.iloc[i]['风险值'],
            'top_3_influences': top_features
        })

    # 统计（已降序）
    all_top_features = [feat for result in results for feat, _ in result['top_3_influences']]
    feature_counts = Counter(all_top_features)

    return results, dict(sorted(feature_counts.items(), key=lambda x: x[1], reverse=True))  # 返回降序dict