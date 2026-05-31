"""Domain registry. Add a domain by creating domains/<name>.py with a SPEC and listing it here."""
import importlib

_KNOWN = ("it", "legal", "marketing")


def get(name: str):
    if name not in _KNOWN:
        raise ValueError(f"unknown domain '{name}'; known: {_KNOWN}")
    return importlib.import_module(f"domains.{name}").SPEC


def available():
    out = []
    for n in _KNOWN:
        try:
            importlib.import_module(f"domains.{n}")
            out.append(n)
        except ModuleNotFoundError:
            pass
    return out
