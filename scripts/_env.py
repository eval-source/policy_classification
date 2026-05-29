"""Shared env bootstrap: load .env and map THINKING_MACHINE_API_KEY -> TINKER_API_KEY."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    # Tinker SDK expects TINKER_API_KEY
    if "TINKER_API_KEY" not in os.environ and "THINKING_MACHINE_API_KEY" in os.environ:
        os.environ["TINKER_API_KEY"] = os.environ["THINKING_MACHINE_API_KEY"]


load_env()
