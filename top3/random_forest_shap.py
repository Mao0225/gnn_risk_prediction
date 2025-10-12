import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
from collections import Counter

def compute_random_forest_shap(df):
    # 定义特征列
    feature_cols = [col for col in df.columns if col not in ['Id', '风险值','Object','FID','属于']]

    # 准备X和y
    X = df[feature_cols]
    y = df['风险值']

    # 标准化
    use_scaling = True
    X_for_model = X
    if use_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_for_model = X_scaled

    # 拟合随机森林
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_for_model, y)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_for_model)

    # 提取前三
    feature_names = feature_cols
    results = []
    for i in range(len(df)):
        grid_contribs = np.abs(shap_values[i])
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