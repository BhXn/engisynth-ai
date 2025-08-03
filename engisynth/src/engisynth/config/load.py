import yaml
from .schema import ProjectCfg

def load_config(path: str) -> ProjectCfg:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ProjectCfg(**data)

