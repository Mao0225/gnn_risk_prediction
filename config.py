import yaml

def load_config(file_path='config.yaml'):
    with open(file_path, 'r', encoding='utf-8') as f:  # 添加 encoding='utf-8'
        return yaml.safe_load(f)