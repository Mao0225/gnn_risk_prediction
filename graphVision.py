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
from pyproj import Transformer

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


# ---------------------- 保存与可视化工具（UTM 50N专用） ----------------------
def _save_graph_and_visualization(G, polygons, pyg_graph, exp_dir):
    os.makedirs(os.path.join(exp_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    # 保存图数据
    torch.save(pyg_graph, os.path.join(exp_dir, "data", "graph.pt"))
    with open(os.path.join(exp_dir, "data", "graph.pkl"), "wb") as f:
        pickle.dump(G, f)
    print("PyG 和 NetworkX 图保存完成")

    # 坐标转换：UTM 50N（EPSG:32650）→ WGS84（EPSG:4326）
    # 已验证此转换可将点正确显示在沈阳附近
    transformer = Transformer.from_crs(
        crs_from="EPSG:32650",  # 关键修复：使用UTM 50N分区
        crs_to="EPSG:4326",
        always_xy=True
    )

    # 转换所有节点中心点坐标
    centroids = []
    for poly in polygons:
        utm_x, utm_y = poly.centroid.x, poly.centroid.y
        lon, lat = transformer.transform(utm_x, utm_y)
        centroids.append((lat, lon))

    # 验证坐标是否在沈阳区域
    print("\n坐标验证结果：")
    in_shenyang = 0
    for i, (lat, lon) in enumerate(centroids[:5]):  # 显示前5个点验证
        if 41.5 < lat < 43.5 and 122 < lon < 124:
            in_shenyang += 1
            print(f"节点{i}: 经纬度({lat:.4f}, {lon:.4f}) → 位于沈阳区域内")
        else:
            print(f"节点{i}: 经纬度({lat:.4f}, {lon:.4f}) → 位于沈阳区域外")
    if len(centroids) > 5:
        print(f"... 共{len(centroids)}个节点，{in_shenyang}/{len(centroids)}位于沈阳区域内")

    # 计算地图中心（确保在沈阳区域）
    center_lat = np.mean([p[0] for p in centroids])
    center_lon = np.mean([p[1] for p in centroids])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)  # 适当放大

    # 添加节点
    for i, (lat, lon) in enumerate(centroids):
        neighbors = list(G.neighbors(i))
        feature_str = ", ".join([f"{v:.2f}" for v in pyg_graph.x[i].numpy()])
        popup_text = f"节点 {i}<br>特征: {feature_str}<br>邻居: {neighbors}<br>经纬度: {lat:.6f}, {lon:.6f}"

        # 节点颜色：沈阳区域内为蓝色，外为橙色
        color = 'blue' if (41.5 < lat < 43.5 and 122 < lon < 124) else 'orange'

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    # 添加边
    for u, v in G.edges():
        coord1 = centroids[u]
        coord2 = centroids[v]
        folium.PolyLine(locations=[coord1, coord2], color='red', weight=1.5).add_to(m)

    # 添加沈阳区域参考框
    folium.Rectangle(
        bounds=[[41.5, 122], [43.5, 124]],
        color='green',
        weight=2,
        fill=False,
        tooltip='沈阳大致范围'
    ).add_to(m)

    # 保存地图
    folium_path = os.path.join(exp_dir, "plots", "graph_map_shenyang.html")
    m.save(folium_path)
    print(f"\n沈阳区域地图保存完成: {folium_path}")

    # 保存JSON数据
    graph_data = {
        "nodes": [{"id": i, "lat": centroids[i][0], "lon": centroids[i][1],
                   "feature": pyg_graph.x[i].tolist(),
                   "neighbors": list(G.neighbors(i))} for i in range(len(polygons))],
        "edges": [{"u": u, "v": v} for u, v in G.edges()]
    }
    json_path = os.path.join(exp_dir, "data", "graph_data_shenyang.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)
    print(f"沈阳区域图数据JSON保存完成: {json_path}")


# ---------------------- 实验目录创建 ----------------------
def create_exp_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_shenyang_utm50n_{timestamp}")
    for sub_dir in ["plots", "data", "model"]:
        os.makedirs(os.path.join(exp_dir, sub_dir), exist_ok=True)
    return exp_dir


# ---------------------- 通用工具函数 ----------------------
def save_json(data, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_numpy(data, save_path):
    np.save(save_path, data)


# ---------------------- Excel 数据加载 ----------------------
def load_and_preprocess_data(file_path, exp_dir=None):
    print(f"开始加载数据文件: {file_path}")
    df = pd.read_excel(file_path)
    print(f"数据加载完成，共 {len(df)} 条记录")

    # 特征列
    feature_columns = ['年均降', '土壤渗', '人口密', '暴雨天', '村庄分',
                       '耕地占', '坡度', 'dem', '医院', '公安',
                       '河网密', '行洪能', '委占比', '弱占比',
                       '卫占比', 'GDP1占比']
    print(f"提取特征列: {feature_columns}")
    features = df[feature_columns].fillna(0).values
    labels = df['风险值'].fillna(0).values

    # 坐标解析（处理三维嵌套格式）
    def safe_parse(x):
        if isinstance(x, str):
            try:
                coord_list = ast.literal_eval(x)
                # 处理 [[[x1,y1], [x2,y2], ...]]] 格式
                if isinstance(coord_list, list) and len(coord_list) > 0:
                    if isinstance(coord_list[0], list) and len(coord_list[0]) > 0:
                        if isinstance(coord_list[0][0], list):
                            return coord_list[0]  # 三维转二维
                        return coord_list  # 已为二维
                return None
            except Exception as e:
                print(f"坐标解析失败: {e}, 原始字符串: {x[:50]}...")
                return None
        return None

    # 解析并过滤无效坐标
    coords = df['Object'].apply(safe_parse)
    valid_idx = coords.notnull()
    coords = coords[valid_idx]
    features = features[valid_idx]
    labels = labels[valid_idx]
    print(f"解析后有效坐标数量: {len(coords)}")

    # 创建多边形
    polygons = []
    for coord in coords:
        try:
            if len(coord) >= 3:
                poly = Polygon(coord)
                polygons.append(poly)
            else:
                print(f"跳过无效多边形（点数不足3）: {coord[:3]}...")
        except Exception as e:
            print(f"创建多边形失败: {e}, 坐标: {coord[:3]}...")

    print(f"最终有效多边形数量: {len(polygons)}")
    if len(polygons) == 0:
        raise ValueError("无有效多边形数据，无法构建图！")

    # 构建图
    data = build_graph_fast(polygons, features[:len(polygons)], exp_dir)
    data.y = torch.tensor(labels[:len(polygons)], dtype=torch.float)
    return data, polygons


# ---------------------- 命令行调用入口 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="沈阳区域 - UTM 50N坐标专用图处理工具")
    parser.add_argument("file_path", type=str, help="Excel文件路径（含Object坐标列）")
    parser.add_argument("--output_dir", type=str, default=None, help="自定义输出目录（可选）")
    args = parser.parse_args()

    exp_dir = args.output_dir or create_exp_dir("results_shenyang")
    print(f"实验结果将保存到: {exp_dir}")

    data, polygons = load_and_preprocess_data(args.file_path, exp_dir)
    print(f"\n✅ 图构建完成！")
    print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
    print(f"地图文件: {os.path.join(exp_dir, 'plots', 'graph_map_shenyang.html')}")
