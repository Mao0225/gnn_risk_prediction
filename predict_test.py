import matplotlib
matplotlib.use('Agg', force=True)  # 必须在所有导入之前

import os
import argparse
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import subprocess
import sys

# 恢复 Polygon 导入（用于处理几何数据构建图结构，非绘图）
from shapely.geometry import Polygon

from model import GNN
from utils import build_graph

def load_and_preprocess_data(file_path):
    """
    加载并预处理Excel格式的地理空间数据
    包含严格的数据校验，确保几何数据有效性，避免后续处理崩溃
    """
    print(f"🔄 正在加载数据: {file_path}")
    try:
        # 加载原始数据
        df = pd.read_excel(file_path)
        print(f"✅ 数据加载成功，共 {len(df)} 行")

        # 定义特征列（统一管理，避免重复书写）
        FEATURE_COLS = [
            '年均降', '土壤渗', '人口密', '暴雨天', '村庄分',
            '耕地占', '坡度', 'dem', '医院', '公安',
            '河网密', '行洪能', '委占比', '弱占比',
            '卫占比', 'GDP1占比'
        ]

        # 检查特征列是否存在
        missing_features = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_features:
            raise ValueError(f"❌ 数据中缺少必要的特征列: {', '.join(missing_features)}")

        # 处理几何数据（多重校验，过滤无效坐标）
        print("🔄 处理几何数据...")
        if 'Object' not in df.columns:
            raise ValueError("❌ 数据中缺少'Object'列（几何坐标数据）")

        coords = []
        valid_indices = []  # 记录有效数据的原索引
        for idx, obj in enumerate(df['Object']):
            try:
                # 1. 跳过空值/非字符串格式
                if pd.isna(obj) or not isinstance(obj, str):
                    print(f"⚠️ 跳过空/非字符串坐标（行{idx}）")
                    continue

                # 2. 解析坐标（捕获语法错误）
                try:
                    coord = eval(obj)
                except (SyntaxError, NameError) as e:
                    print(f"⚠️ 坐标格式解析失败（行{idx}）: {str(e)}, 数据: {obj[:50]}...")
                    continue

                # 3. 校验坐标基本格式（必须是非空列表）
                if not isinstance(coord, list) or len(coord) == 0:
                    print(f"⚠️ 坐标不是有效列表（行{idx}）: {obj[:50]}...")
                    continue

                # 4. 提取多边形点列表（兼容嵌套格式）
                polygon_points = coord[0] if (isinstance(coord[0], list) and
                                              all(isinstance(p, (list, tuple)) for p in coord[0])) else coord

                # 5. 校验多边形点数（至少3个点才能构成多边形）
                if len(polygon_points) < 3:
                    print(f"⚠️ 多边形点数不足3个（行{idx}）: 实际{len(polygon_points)}个点")
                    continue

                # 6. 校验每个点是否为二维有效数字
                valid_point = True
                for point in polygon_points:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        print(f"⚠️ 无效坐标点（行{idx}）: {point}")
                        valid_point = False
                        break
                    # 校验坐标是否为数字
                    try:
                        float(point[0])
                        float(point[1])
                    except (ValueError, TypeError):
                        print(f"⚠️ 坐标不是有效数字（行{idx}）: {point}")
                        valid_point = False
                        break

                if valid_point:
                    coords.append(coord)
                    valid_indices.append(idx)

            except Exception as e:
                print(f"⚠️ 处理坐标时发生未知错误（行{idx}）: {str(e)}")
                continue

        # 检查是否有有效坐标
        if len(coords) == 0:
            raise ValueError("❌ 无任何有效几何坐标数据，无法继续处理")

        # 筛选有效数据行
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ 已筛选出有效数据 {len(df_valid)} 行（原始数据 {len(df)} 行）")

        # 创建有效多边形（过滤面积为0的无效多边形）
        try:
            polygons = []
            for coord in coords:
                poly_points = coord[0] if isinstance(coord[0], list) else coord
                poly = Polygon(poly_points)
                # 过滤面积接近0的多边形（避免后续处理崩溃）
                if poly.area > 1e-8:
                    polygons.append(poly)
            print(f"✅ 成功创建 {len(polygons)} 个有效多边形（过滤无效多边形 {len(coords)-len(polygons)} 个）")
        except Exception as e:
            raise ValueError(f"❌ 创建多边形时出错: {str(e)}")

        # 提取匹配的特征数据（确保与有效多边形数量一致）
        features = df[FEATURE_COLS].iloc[valid_indices].values[:len(polygons)]
        print(f"✅ 提取特征数据: {features.shape[0]}行, {features.shape[1]}列")

        # 处理标签（确保与有效数据匹配）
        labels = None
        if '风险值' in df_valid.columns:
            labels = df_valid['风险值'].values[:len(polygons)]  # 与多边形数量对齐
            print(f"✅ 发现真实标签，共 {len(labels)} 个（与多边形数量一致）")
        else:
            print("ℹ️ 数据中未包含'风险值'列，仅进行预测")

        # 构建图结构
        print("🔄 构建图结构...")
        try:
            data = build_graph(polygons, features)
            print(f"✅ 图构建完成，节点数: {data.x.shape[0]}, 边数: {data.edge_index.shape[1]}")
        except Exception as e:
            raise RuntimeError(f"❌ 构建图结构失败: {str(e)}")

        # 绑定标签（确保数量一致）
        if labels is not None:
            if len(labels) != data.x.shape[0]:
                raise ValueError(f"❌ 标签数量（{len(labels)}）与节点数量（{data.x.shape[0]}）不匹配")
            data.y = torch.tensor(labels, dtype=torch.float)

        return df_valid, data, polygons

    except Exception as e:
        print(f"❌ 数据预处理失败: {str(e)}")
        raise  # 向上层抛出异常，便于定位问题


def predict(file_path, model_path, base_out_dir):
    print("🚀 开始预测流程...")
    # 1. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_out_dir, f"predict_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"📁 输出目录: {exp_dir}")

    try:
        # 2. 加载数据（已做严格校验）
        df_valid, data, polygons = load_and_preprocess_data(file_path)

        # 3. 加载模型
        print(f"🔄 加载模型: {model_path}")
        model = GNN(in_channels=data.x.shape[1], hidden_channels=128)
        # 兼容PyTorch 2.0+，避免安全警告
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        print("✅ 模型加载成功")

        # 4. 预测（去除多余维度，确保与数据匹配）
        print("🔄 开始预测...")
        with torch.no_grad():
            preds = model(data).numpy().squeeze()  # 挤压多余维度
        print(f"✅ 预测完成，预测值范围: {preds.min():.3f} - {preds.max():.3f}")

        # 确保预测值与有效数据数量一致
        if len(preds) != len(df_valid):
            print(f"⚠️ 预测值数量({len(preds)})与有效数据行数({len(df_valid)})不匹配，将截断/补全")
            preds = preds[:len(df_valid)] if len(preds) > len(df_valid) else np.pad(preds, (0, len(df_valid)-len(preds)))

        # 5. 保存结果
        df_result = df_valid.copy()




        df_result["预测风险值"] = preds
        if "风险值" in df_result.columns:
            df_result["差值"] = df_result["风险值"] - df_result["预测风险值"]
            df_result["绝对差值"] = np.abs(df_result["差值"])

        excel_path = os.path.join(exp_dir, "predict_results.xlsx")
        df_result.to_excel(excel_path, index=False)
        print(f"✅ 预测结果已保存到: {excel_path}")
        # 🔄 调用独立绘图脚本 (避免和 torch 冲突)
        try:
            subprocess.run(
                [sys.executable, "predict_plot.py", excel_path, exp_dir],
                check=True
            )
            print("✅ 可视化图表已生成")
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")

        # 6. 计算评估指标（仅当有真实标签时）
        if "风险值" in df_result.columns:
            true_values = df_result["风险值"].values
            mse = np.mean((true_values - preds) ** 2)
            mae = np.mean(np.abs(true_values - preds))
            rmse = np.sqrt(mse)
            # 避免R²计算时分母为0
            ss_total = np.sum((true_values - np.mean(true_values)) ** 2)
            r2 = 1 - np.sum((true_values - preds) ** 2) / ss_total if ss_total != 0 else 0.0

            print("📊 评估指标:")
            print(f"   MSE:  {mse:.4f}")
            print(f"   MAE:  {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   R²:   {r2:.4f}")

            # 保存指标
            metrics_df = pd.DataFrame([{"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}])
            metrics_path = os.path.join(exp_dir, "metrics.xlsx")
            metrics_df.to_excel(metrics_path, index=False)
            print(f"📊 指标已保存到: {metrics_path}")

        # 7. 列出生成的文件
        print("\n📋 生成的文件:")
        files = sorted(os.listdir(exp_dir))
        if not files:
            print("   ⚠️ 未生成任何文件")
        else:
            for file in files:
                file_path = os.path.join(exp_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"   📄 {file} ({size} bytes)")

        print(f"\n🎉 预测完成! 所有结果保存在: {exp_dir}")

    except Exception as e:
        print(f"❌ 预测过程出错: {str(e)}")
        import traceback
        traceback.print_exc()  # 输出完整错误栈


if __name__ == "__main__":
    print("🔥 程序启动...")

    try:
        parser = argparse.ArgumentParser(description="GNN风险值预测工具")
        parser.add_argument("--data", type=str, help="输入Excel数据文件路径", default="data/test.xlsx")
        parser.add_argument("--model", type=str, default="models/best_model.pth", help="训练好的模型路径")
        parser.add_argument("--out", type=str, default="results/predict", help="输出目录")
        args = parser.parse_args()

        print(f"   参数:")
        print(f"   数据文件: {args.data}")
        print(f"   模型文件: {args.model}")
        print(f"   输出目录: {args.out}")

        # 验证文件存在性
        if not os.path.exists(args.data):
            print(f"❌ 数据文件不存在: {args.data}")
            exit(1)
        if not os.path.exists(args.model):
            print(f"❌ 模型文件不存在: {args.model}")
            exit(1)

        # 执行预测
        predict(args.data, args.model, args.out)

    except Exception as e:
        print(f"💥 程序异常终止: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)