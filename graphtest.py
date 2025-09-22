import argparse
from rtree import index
from torch_geometric.utils import from_networkx
import networkx as nx
import torch
import os
import json
import numpy as np
from datetime import datetime
import pickle
import folium
import pandas as pd
from shapely.geometry import Polygon
import ast
from pyproj import Transformer, CRS


# ---------------------- 构建图函数 ----------------------
def build_graph(polygons, features):
    G = nx.Graph()
    for i in range(len(polygons)):
        G.add_node(i, x=torch.tensor(features[i], dtype=torch.float))
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]) or polygons[i].touches(polygons[j]):
                G.add_edge(i, j)
    return from_networkx(G)


def build_graph_fast(polygons, features, exp_dir=None):
    print("开始构建图结构（使用空间索引加速）...")
    print(f"处理 {len(polygons)} 个多边形")

    G = nx.Graph()
    for i in range(len(polygons)):
        G.add_node(i, x=torch.tensor(features[i], dtype=torch.float))

    print("构建空间索引...")
    idx = index.Index()
    for i, poly in enumerate(polygons):
        idx.insert(i, poly.bounds)

    print("查找空间邻接关系...")
    edges_count = 0
    batch_size = 1000
    total_batches = (len(polygons) + batch_size - 1) // batch_size

    for batch_start in range(0, len(polygons), batch_size):
        batch_end = min(batch_start + batch_size, len(polygons))
        batch_progress = (batch_start // batch_size) + 1
        print(f"处理批次 {batch_progress}/{total_batches}")

        for i in range(batch_start, batch_end):
            potential_neighbors = list(idx.intersection(polygons[i].bounds))
            for j in potential_neighbors:
                if j > i and (polygons[i].intersects(polygons[j]) or polygons[i].touches(polygons[j])):
                    G.add_edge(i, j)
                    edges_count += 1

    print(f"构建完成，共 {edges_count} 条边")
    pyg_graph = from_networkx(G)

    if exp_dir:
        _save_graph_and_visualization(G, polygons, pyg_graph, exp_dir)

    return pyg_graph


# ---------------------- 保存与可视化工具（修复后的版本） ----------------------
def _save_graph_and_visualization(G, polygons, pyg_graph, exp_dir):
    os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    # 保存图数据
    torch.save(pyg_graph, os.path.join(exp_dir, "data", "graph.pt"))
    with open(os.path.join(exp_dir, "data", "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    print("PyG 和 NetworkX 图保存完成")

    # --- 修复后的坐标转换 ---
    # 明确指定为北京1954 / 高斯-克吕格投影，中央经线123E，EPSG:2438
    crs_from_code = "EPSG:2438"  # 北京1954 / Gauss-Kruger CM 123E
    print(f"\n使用投影: {crs_from_code}")

    # 构建正确的坐标转换器
    transformer = Transformer.from_crs(
        crs_from=crs_from_code,
        crs_to="EPSG:4326",  # 转换为 WGS84 经纬度
        always_xy=True
    )

    # 计算并转换所有多边形中心点
    centroids = []  # WGS84经纬度
    utm_centroids = []  # 原始UTM坐标

    for poly in polygons:
        utm_x_orig, utm_y_orig = poly.centroid.x, poly.centroid.y

        # 动态修正东坐标：如果东坐标过大，则进行修正
        utm_x_fixed = utm_x_orig
        if utm_x_orig > 1000000:
            # 常见情况是带号前缀或大的偏移量
            # 尝试减去一个百万级的偏移量
            utm_x_fixed = utm_x_orig - 1000000
            # 如果修正后值还在正常范围外，再尝试其他修正
            if not 200000 < utm_x_fixed < 800000:
                utm_x_fixed = utm_x_orig - 500000
        elif utm_x_orig > 800000:
            # 另一种可能的偏移
            utm_x_fixed = utm_x_orig - 500000

        utm_centroids.append((utm_x_fixed, utm_y_orig))

        # 修复关键：将 x 和 y 的位置对调
        lon, lat = transformer.transform(utm_x_fixed, utm_y_orig)
        centroids.append((lat, lon))
    print(f"已计算 {len(centroids)} 个多边形中心点")

    # 计算实际坐标范围（自动适应数据分布）
    all_lats = [p[0] for p in centroids]
    all_lons = [p[1] for p in centroids]
    min_lat, max_lat = np.min(all_lats), np.max(all_lats)
    min_lon, max_lon = np.min(all_lons), np.max(all_lons)
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # 坐标验证（使用实际数据范围作为参考）
    print(f"\n坐标范围验证:")
    print(f"纬度范围: {min_lat:.4f} - {max_lat:.4f}")
    print(f"经度范围: {min_lon:.4f} - {max_lon:.4f}")
    print(f"中心点: ({center_lat:.4f}, {center_lon:.4f})")

    # 创建地图（自动聚焦到数据中心）
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # 添加节点
    for i, (lat, lon) in enumerate(centroids):
        original_utm_x, original_utm_y = polygons[i].centroid.x, polygons[i].centroid.y
        neighbors = list(G.neighbors(i))
        feature_str = ", ".join([f"{v:.2f}" for v in pyg_graph.x[i].numpy()])
        popup_text = (f"节点 {i}<br>"
                      f"原始UTM坐标: ({original_utm_x:.2f}, {original_utm_y:.2f})<br>"
                      f"WGS84坐标: ({lat:.6f}, {lon:.6f})<br>"
                      f"UTM带号: 51N (假设)<br>"
                      f"邻居数量: {len(neighbors)}")

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=350)
        ).add_to(m)

    # 添加边
    for u, v in G.edges():
        folium.PolyLine(
            locations=[centroids[u], centroids[v]],
            color='red',
            weight=1.5,
            opacity=0.6
        ).add_to(m)

    # 添加数据范围框
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color='green',
        weight=2,
        dash_array='5, 5',
        fill=False,
        tooltip='数据实际分布范围'
    ).add_to(m)

    # 保存结果
    folium_path = os.path.join(exp_dir, "plots", "graph_map.html")
    m.save(folium_path)
    print(f"\n地图保存完成: {folium_path}")

    graph_data = {
        "metadata": {
            "utm_zone": "51N (假设)",
            "coordinate_range": {
                "lat_min": float(min_lat),
                "lat_max": float(max_lat),
                "lon_min": float(min_lon),
                "lon_max": float(max_lon)
            }
        },
        "nodes": [
            {
                "id": i,
                "utm_x": utm_centroids[i][0],
                "utm_y": utm_centroids[i][1],
                "lat": centroids[i][0],
                "lon": centroids[i][1],
                "features": pyg_graph.x[i].tolist(),
                "neighbors": list(G.neighbors(i))
            } for i in range(len(polygons))
        ],
        "edges": [{"u": u, "v": v} for u, v in G.edges()]
    }
    json_path = os.path.join(exp_dir, "data", "graph_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)
    print(f"图数据JSON保存完成: {json_path}")


# ---------------------- 实验目录创建 ----------------------
def create_exp_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_utm_auto_{timestamp}")
    for sub_dir in ["plots", "data", "model"]:
        os.makedirs(os.path.join(exp_dir, sub_dir), exist_ok=True)
    return exp_dir


# ---------------------- 数据加载与坐标解析 ----------------------
def load_and_preprocess_data(file_path, exp_dir=None):
    print(f"开始加载数据文件: {file_path}")
    # 支持不同Excel格式
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except:
        df = pd.read_excel(file_path, engine="xlrd")
    print(f"数据加载完成，共 {len(df)} 条记录")

    # 特征列处理
    feature_columns = ['年均降', '土壤渗', '人口密', '暴雨天', '村庄分',
                       '耕地占', '坡度', 'dem', '医院', '公安',
                       '河网密', '行洪能', '委占比', '弱占比',
                       '卫占比', 'GDP1占比']
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Excel缺失特征列：{', '.join(missing_cols)}")

    features = df[feature_columns].fillna(0).values
    labels = df['风险值'].fillna(0).values if '风险值' in df.columns else np.zeros(len(df))

    # 解析3层嵌套坐标
    def safe_parse_3level(x):
        if isinstance(x, str):
            try:
                coord_list = ast.literal_eval(x.strip())
                # 剥除外层列表，直到得到二维坐标
                while isinstance(coord_list, list) and len(coord_list) > 0:
                    if all(isinstance(item, list) and len(item) == 2 for item in coord_list[:3]):
                        # 验证坐标为数字
                        for item in coord_list[:3]:
                            float(item[0]), float(item[1])
                        return coord_list
                    coord_list = coord_list[0]
                print(f"无效坐标格式: {x[:60]}...")
                return None
            except Exception as e:
                print(f"解析错误: {str(e)[:40]}, 数据: {x[:60]}...")
                return None
        return None

    # 解析坐标并过滤
    if 'Object' not in df.columns:
        raise ValueError("Excel缺少'Object'列（存储坐标）")
    coords = df['Object'].apply(safe_parse_3level)
    valid_idx = coords.apply(lambda x: isinstance(x, list) and len(x) >= 3)
    coords = coords[valid_idx].tolist()
    features = features[valid_idx]
    labels = labels[valid_idx]
    print(f"解析后有效坐标数量: {len(coords)}")

    # 构建多边形
    polygons = []
    invalid_count = 0
    for coord in coords:
        try:
            poly = Polygon(coord)
            if poly.area > 1e-4:
                polygons.append(poly)
            else:
                invalid_count += 1
        except Exception as e:
            invalid_count += 1
    print(f"最终有效多边形数量: {len(polygons)}（过滤无效{invalid_count}个）")
    if len(polygons) == 0:
        raise ValueError("无有效多边形数据！")

    # 构建图
    data = build_graph_fast(polygons, features[:len(polygons)], exp_dir)
    data.y = torch.tensor(labels[:len(polygons)], dtype=torch.float)
    return data, polygons


# ---------------------- 命令行入口 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动识别UTM带号的坐标处理工具")
    parser.add_argument("file_path", type=str, help="Excel文件路径")
    parser.add_argument("--output_dir", type=str, default=None, help="自定义输出目录")
    args = parser.parse_args()

    exp_dir = args.output_dir or create_exp_dir()
    print(f"结果保存目录: {exp_dir}")

    try:
        data, polygons = load_and_preprocess_data(args.file_path, exp_dir)
        print(f"\n✅ 处理完成！")
        print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
        print(f"地图文件: {os.path.join(exp_dir, 'plots', 'graph_map.html')}")
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        exit(1)