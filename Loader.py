import pandas as pd
from pathlib import Path
import yaml


def read_yaml(file_path):
    with open(file_path, encoding="utf-8") as file:
        return pd.DataFrame(yaml.safe_load(file))


pd.read_yaml = read_yaml


def read(file_path):
    ext = Path(file_path).suffix.lower()[1:]
    return getattr(pd, f"read_{ext}")(file_path)
