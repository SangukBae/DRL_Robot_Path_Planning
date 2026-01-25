"""Path utilities for dataset directory structure."""
import os
from datetime import datetime
import pytz


def generate_run_id(prefix: str = "run") -> str:
    """Generate run_id with Asia/Seoul timezone.

    Returns:
        str: run_YYYY-MM-DD_HH-MM-SS
    """
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    return f"{prefix}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def format_segment_id(index: int, width: int = 4, prefix: str = "seg") -> str:
    """Format segment id as seg_XXXX with zero padding.

    Args:
        index (int): 1-based segment index (recommended). If 0 is given, it will be allowed.
        width (int): Zero padding width (default: 4 -> XXXX).
        prefix (str): Segment prefix (default: "seg").

    Returns:
        str: segment identifier (e.g., "seg_0001")
    """
    if not isinstance(index, int):
        raise TypeError(f"index must be int, got {type(index)}")
    if index < 0:
        raise ValueError("index must be >= 0")
    if width < 1:
        raise ValueError("width must be >= 1")

    return f"{prefix}_{index:0{width}d}"


def get_run_dir(dataset_root: str, run_id: str) -> str:
    """Get run directory path."""
    return os.path.join(dataset_root, "runs", run_id)


def get_segments_dir(dataset_root: str, run_id: str) -> str:
    """Get segments directory path."""
    return os.path.join(get_run_dir(dataset_root, run_id), "segments")


def get_segment_dir(dataset_root: str, run_id: str, segment_id: str) -> str:
    """Get specific segment directory path (segment_id should be like 'seg_0001')."""
    return os.path.join(get_segments_dir(dataset_root, run_id), segment_id)


def get_segment_dir_by_index(dataset_root: str, run_id: str, index: int, width: int = 4) -> str:
    """Get segment directory path by integer index (consistent seg_XXXX formatting)."""
    segment_id = format_segment_id(index=index, width=width)
    return get_segment_dir(dataset_root, run_id, segment_id)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)