from pathlib import Path
from typing import Union, Dict, Any
import json
import re
import yaml
import pandas as pd


def write_generated_code(code: str, output_path: Union[str, Path]) -> None:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Generated code is empty or invalid")

    path = Path(output_path)

    if not path.suffix == ".py":
        raise ValueError("Output path must be a .py file")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(code)

def write_generated_yaml(spec: Dict[str, Any], output_path: Union[str, Path]) -> None:
    if not isinstance(spec, dict) or not spec:
        raise ValueError("Generated spec is empty or invalid")

    path = Path(output_path)

    if path.suffix != ".yaml":
        raise ValueError("Output path must be a .yaml file")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

def load_schema(name: str) -> dict:
    path = Path("schemas") / name
    
    try:
        with path.open() as f:
            return json.load(f)
    except:
        raise FileNotFoundError(f"Path {path} does not exist")

def load_spec(name: str) -> dict:
    path = Path("specs") / name
    
    try:
        with path.open() as f:
            return yaml.safe_load(f)
    except:
        raise FileNotFoundError(f"Path {path} does not exist")

def extract_content_from_response(text: str) -> str:
    # Find ```yaml fenced block
    yaml_block = re.search(r"```(?:yaml|yml)?\n(.*?)```", text, re.DOTALL)
    if yaml_block:
        return yaml_block.group(1)

    python_block = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if python_block:
        return python_block.group(1)
    
    # Fallback: any fenced block
    generic_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_block:
        return generic_block.group(1)

    return text.strip()

def build_dataframe(x, y):
    return pd.DataFrame({
        "x": x,
        "y": y,
    })

def save_dataset_csv(df: pd.DataFrame, output_path: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame is empty")
    
    path = Path(output_path)

    if path.suffix != ".csv":
        raise ValueError("Output path must be a .csv file")

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

def load_dataset_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix != ".csv":
        raise ValueError("Dataset must be a .csv file")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    return df