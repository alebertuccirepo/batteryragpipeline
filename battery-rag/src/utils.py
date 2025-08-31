from pathlib import Path
import yaml

def load_config(path: str | Path = "rag_config.yaml") -> dict:
    """Load YAML config from project root."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | Path) -> Path:
    """Create directory if missing; return Path."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
