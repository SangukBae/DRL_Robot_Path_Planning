"""YAML I/O utilities."""
import os
import yaml
from .paths import ensure_dir


def save_yaml(data, filepath: str) -> None:
    """Save data to YAML file with UTF-8 encoding.

    Args:
        data: Data to save (typically dict/list)
        filepath (str): Output file path
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        ensure_dir(dirpath)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )


def load_yaml(filepath: str):
    """Load YAML file with UTF-8 encoding.

    - If the file is empty, returns {} instead of None.
    - Raises a RuntimeError with a clearer message on parse errors / missing file.

    Args:
        filepath (str): Input file path

    Returns:
        dict | list: Loaded data, or {} for empty content
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except FileNotFoundError as e:
        raise RuntimeError(f"YAML file not found: {filepath}") from e
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML file: {filepath}") from e
