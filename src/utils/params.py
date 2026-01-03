import yaml
from pathlib import Path

def load_params():
    config_path = Path("src/config/params.yaml")
    with open(config_path) as f:
        params = yaml.safe_load(f)
    return params

params = load_params()
