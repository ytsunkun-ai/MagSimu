"""File-system, JSON and provenance helpers.

The project treats the file system as the stable Python-Julia interface.  Every
run writes its own directory with a manifest, so a result can always be traced
back to the exact configuration, random seed and software versions.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib.metadata as _metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:  # PyYAML is a declared dependency; the fallback makes error messages nicer.
    import yaml
except Exception:  # pragma: no cover - only used in incomplete environments
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def project_root(start: str | Path | None = None) -> Path:
    """Locate the repository root (the directory containing ``pyproject.toml``)."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # Editable installs: python/softmag/io_utils.py -> repository root
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a user-supplied path relative to *base* (default: cwd)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return Path(base or Path.cwd()).resolve() / p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_utc() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def utc_now_for_filename() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{utc_now_for_filename()}"


def new_run_dir(base: str | Path, run_id: str | None = None) -> tuple[Path, str]:
    run_id = run_id or make_run_id()
    out = ensure_dir(Path(base) / run_id)
    return out, run_id


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:  # pragma: no cover
            pass
    if pd.isna(value):  # scalar NaN / NaT
        return None
    return str(value)


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas / pydantic objects to JSON-safe data."""
    return _jsonable(value)


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    text = json.dumps(
        to_jsonable(payload), indent=indent, ensure_ascii=False, allow_nan=False
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# YAML / CSV
# ---------------------------------------------------------------------------
def _require_yaml() -> Any:
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML 未安装，无法读取 YAML 配置：pip install pyyaml")
    return yaml


def read_yaml(path: str | Path) -> Any:
    y = _require_yaml()
    with Path(path).open("r", encoding="utf-8") as fh:
        return y.safe_load(fh)


def write_yaml(path: str | Path, payload: Any) -> Path:
    y = _require_yaml()
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        y.safe_dump(to_jsonable(payload), fh, allow_unicode=True, sort_keys=False)
    return path


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    return path


# ---------------------------------------------------------------------------
# Hashing / versions / manifest
# ---------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def software_versions() -> dict[str, Any]:
    packages = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "pydantic",
        "matplotlib",
        "PyYAML",
        "softmag-predictor",
    ):
        try:
            packages[name] = _metadata.version(name)
        except _metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "julia_version": julia_version(),
    }


def julia_version(executable: str = "julia", timeout_s: int = 10) -> str | None:
    exe = shutil.which(executable) or (executable if Path(executable).is_file() else None)
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    return text or None


def git_revision(start: str | Path | None = None) -> str | None:
    """Return the current git commit if the project happens to be a git repo."""
    root = project_root(start)
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def build_manifest(
    *,
    run_id: str,
    config_paths: Iterable[str | Path] = (),
    config: Any | None = None,
    random_seed: int | None = None,
    notes: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a reproducibility manifest for one run."""
    paths = [Path(p) for p in config_paths]
    files = []
    for p in paths:
        if p.is_file():
            files.append(
                {
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_utc": timestamp_utc(),
        "random_seed": random_seed,
        "software": software_versions(),
        "git_revision": git_revision(),
        "input_files": files,
        "input_config": to_jsonable(config) if config is not None else None,
        "notes": notes,
    }
    if extra:
        manifest["extra"] = to_jsonable(dict(extra))
    return manifest


def relative_to_root(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(project_root()))
    except ValueError:
        return str(path)


def safe_name(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in text.strip()]
    return "".join(keep).strip("_") or "run"


def copy_if_exists(src: str | Path, dst: str | Path) -> Path | None:
    src = Path(src)
    if not src.is_file():
        return None
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` in a nested dict, creating intermediate dicts."""
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = value


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
