from rtree import index
from torch_geometric.utils import from_networkx
import networkx as nx
import torch
import torch_geometric as pyg
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle  # 用于保存 NetworkX 图
import folium  # folium 地图可视化

# ---------------------- 原有构建图函数 ----------------------
def build_graph(polygons, features):
    G = nx.Graph()
    for i in range(len(polygons)):
        G.add_node(i, x=torch.tensor(features[i], dtype=torch.float))
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]) or polygons[i].touches(polygons[j]):
                G.add_edge(i, j)
    return pyg.utils.from_networkx(G)


def build_graph_fast(polygons, features, exp_dir=None):
    print("开始构建图结构（使用空间索引加速）...")
    print(f"处理 {len(polygons)} 个多边形")

    G = nx.Graph()

    # 添加节点
    for i in range(len(polygons)):
        G.add_node(i, x=torch.tensor(features[i], dtype=torch.float))

    # 创建R-tree空间索引
    print("构建空间索引...")
    idx = index.Index()
    for i, poly in enumerate(polygons):
        idx.insert(i, poly.bounds)

    # 使用空间索引加速邻接查询
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
                if j > i:
                    if polygons[i].intersects(polygons[j]) or polygons[i].touches(polygons[j]):
                        G.add_edge(i, j)
                        edges_count += 1

    print(f"构建完成，共 {edges_count} 条边")

    # 转换为 PyG Data 对象
    pyg_graph = from_networkx(G)

    # 如果指定了实验目录，则自动保存
    if exp_dir:
        # 保存 PyG 图
        torch.save(pyg_graph, os.path.join(exp_dir, "data", "graph.pt"))
        print("PyG 图保存完成")

        # 保存 NetworkX 图（使用 pickle 兼容最新版本）
        with open(os.path.join(exp_dir, "data", "graph.pkl"), "wb") as f:
            pickle.dump(G, f)
        print("NetworkX 图保存完成（pickle 格式）")

        # ---------------------- folium 可视化 ----------------------
        try:
            # 地图中心选平均经纬度
            lats = [poly.centroid.y for poly in polygons]
            lons = [poly.centroid.x for poly in polygons]
            center = [sum(lats)/len(lats), sum(lons)/len(lons)]
            m = folium.Map(location=center, zoom_start=12)

            # 添加节点
            for i, poly in enumerate(polygons):
                folium.CircleMarker(
                    location=[poly.centroid.y, poly.centroid.x],
                    radius=4,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    popup=f"节点 {i}"
                ).add_to(m)

            # 添加边
            for u, v in G.edges():
                coord1 = [polygons[u].centroid.y, polygons[u].centroid.x]
                coord2 = [polygons[v].centroid.y, polygons[v].centroid.x]
                folium.PolyLine(locations=[coord1, coord2], color='red', weight=1).add_to(m)

            # 保存 folium 地图 HTML
            folium_path = os.path.join(exp_dir, "plots", "graph_map.html")
            m.save(folium_path)
            print(f"Folium 地图可视化保存完成: {folium_path}")

            # 保存节点和边数据为 JSON
            graph_data = {
                "nodes": [{"id": i, "lat": poly.centroid.y, "lon": poly.centroid.x} for i, poly in enumerate(polygons)],
                "edges": [{"u": u, "v": v} for u, v in G.edges()]
            }
            json_path = os.path.join(exp_dir, "data", "graph_data.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=4)
            print(f"Graph 数据 JSON 保存完成: {json_path}")

        except Exception as e:
            print(f"Folium 可视化失败: {e}")

    return pyg_graph

# ---------------------- 新增保存工具函数 ----------------------
def create_exp_dir(base_dir="results"):
    """
    创建唯一实验目录（含时间戳），避免结果覆盖
    返回：实验目录路径（str）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")
    sub_dirs = ["plots", "data", "model"]
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(exp_dir, sub_dir), exist_ok=True)
    return exp_dir

def save_json(data, save_path):
    """保存JSON格式数据（如配置、评估指标）"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def save_numpy(data, save_path):
    """保存NumPy格式数据（如损失值、预测结果）"""
    np.save(save_path, data)
