import pandas as pd
from shapely.geometry import Polygon
import torch
import ast
from utils import build_graph,build_graph_fast

def load_and_preprocess_data(file_path,exp_dir):
    """
    加载并预处理Excel格式的地理空间数据

    参数:
        file_path: str, Excel数据文件的路径

    返回:
        data: PyG图数据对象，包含节点特征、边索引和标签
        polygons: list, Shapely多边形对象列表，用于可视化
    """
    print(f"开始加载数据文件: {file_path}")

    # 读取Excel数据
    df = pd.read_excel(file_path)
    print(f"数据加载完成，共 {len(df)} 条记录")

    # 提取特征列
    feature_columns = ['年均降', '土壤渗', '人口密', '暴雨天', '村庄分',
                       '耕地占', '坡度', 'dem', '医院', '公安',
                       '河网密', '行洪能', '委占比', '弱占比',
                       '卫占比', 'GDP1占比']

    print(f"提取特征列: {feature_columns}")
    features = df[feature_columns].values
    print(f"特征矩阵形状: {features.shape}")

    # 提取目标变量
    labels = df['风险值'].values
    print(f"标签向量形状: {labels.shape}")

    # 安全解析坐标列
    def safe_parse(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
        elif isinstance(x, (list, tuple)):  # 已经是 list/tuple
            return x
        else:
            return None

    print("解析坐标数据...")
    coords = df['Object'].apply(safe_parse)

    # 过滤掉 None 的情况
    valid_idx = coords.notnull()
    invalid_count = len(coords) - sum(valid_idx)

    if invalid_count > 0:
        print(f"警告: 发现 {invalid_count} 条记录的坐标数据无效")

    coords = coords[valid_idx]
    features = features[valid_idx]
    labels = labels[valid_idx]
    print(f"有效数据记录数: {len(coords)}")

    # 创建多边形
    print("创建多边形对象...")
    polygons = []
    invalid_polygon_count = 0

    for i, coord in enumerate(coords):
        try:
            polygons.append(Polygon(coord[0]))
        except Exception as e:
            invalid_polygon_count += 1
            # 如果格式不对就跳过
            continue

    if invalid_polygon_count > 0:
        print(f"警告: {invalid_polygon_count} 个多边形创建失败")

    print(f"成功创建 {len(polygons)} 个多边形")

    # 构建图结构数据
    print("构建图结构数据...")
    # data = build_graph(polygons, features)
    data = build_graph_fast(polygons, features,exp_dir)

    print(f"图构建完成，节点数: {data.num_nodes}, 边数: {data.num_edges}")

    # 添加标签
    data.y = torch.tensor(labels, dtype=torch.float)
    print("标签已添加到图数据中")

    print("数据预处理完成!")
    return data, polygons