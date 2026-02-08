from pathlib import Path
from typing import Union, Dict, Any
import json
import re
import yaml


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
    

def extract_yaml(llm_output: str) -> str:
    fenced = re.search(r"```yaml\s*(.*?)\s*```", llm_output, re.DOTALL)
    
    if fenced:
        return fenced.group(1).strip()

    # Fallback: assume entire output is YAML
    return llm_output.strip()