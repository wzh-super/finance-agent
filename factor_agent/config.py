"""Central runtime configuration; the LLM client reads credentials from the environment.

Relative YAML paths resolve against the configuration file; defaults use cwd.
Model and trading settings live in templates/ and cannot be changed by the agent.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import yaml


def default_qlib_python() -> str:
    candidate = Path(sys.executable).parent.parent.parent / "rdagent4qlib/bin/python"
    return str(candidate) if candidate.is_file() else sys.executable


@dataclass
class AgentConfig:
    data_dir: Path = Path("git_ignore_folder/factor_implementation_source_data")
    debug_data_dir: Path = Path("git_ignore_folder/factor_implementation_source_data_debug")
    qlib_provider: Path = Path("~/.qlib/qlib_data/cn_data")
    python_executable: str = sys.executable
    qlib_python: str = ""
    max_coding_attempts: int = 10  # Shared batch evolution rounds, one attempt per unfinished factor per round.
    max_task_failures: int = 20
    memory_top_k: int = 1
    memory_recent_failures: int = 3
    factor_timeout: int = 3600
    backtest_timeout: int = 3600
    chat_model: str = ""
    embedding_model: str = ""
    api_base: str = ""
    api_timeout: int = 120
    api_retries: int = 3

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.debug_data_dir = Path(self.debug_data_dir).expanduser().resolve()
        self.qlib_provider = Path(self.qlib_provider).expanduser().resolve()
        self.qlib_python = self.qlib_python or default_qlib_python()
        for name in ("python_executable", "qlib_python"):
            value = os.path.expanduser(getattr(self, name))
            setattr(self, name, str(Path(value).resolve()) if "/" in value else shutil.which(value) or value)
        self.chat_model = self.chat_model or os.environ.get("CHAT_MODEL", "gpt-4-turbo")
        self.embedding_model = self.embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_base = self.api_base or os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        for name in ("max_coding_attempts", "max_task_failures", "memory_top_k", "memory_recent_failures",
                     "factor_timeout", "backtest_timeout", "api_timeout", "api_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_file(cls, path: Path | None = None):
        if path is None:
            return cls()
        path = path.expanduser().resolve()
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(content, dict):
            raise ValueError("Configuration must be a YAML mapping")
        unknown = set(content) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for name in ("data_dir", "debug_data_dir", "qlib_provider"):
            if name in content:
                value = Path(content[name]).expanduser()
                content[name] = value if value.is_absolute() else path.parent / value
        for name in ("python_executable", "qlib_python"):
            if name in content and "/" in content[name]:
                value = Path(content[name]).expanduser()
                content[name] = str(value if value.is_absolute() else path.parent / value)
        return cls(**content)

    def public_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}

    def fingerprint(self) -> str:
        """Changing data, model, templates or execution settings invalidates a resume."""
        from .execution import data_fingerprint, runtime_fingerprint
        values = self.public_dict()
        assets = Path(__file__).parent
        values["assets"] = {p.relative_to(assets).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in ("prompts", "templates")
                            for p in sorted((assets / folder).rglob("*")) if p.suffix in {".txt", ".yaml"}}
        values["source"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(assets.glob("*.py"))}
        values["data"] = {name: data_fingerprint(getattr(self, name))
                          for name in ("data_dir", "debug_data_dir", "qlib_provider")}
        values["runtimes"] = {name: runtime_fingerprint(getattr(self, name))
                              for name in ("python_executable", "qlib_python")}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def validate_paths(self):
        for directory in (self.data_dir, self.debug_data_dir):
            if not (directory / "daily_pv.h5").is_file():
                raise FileNotFoundError(f"Missing factor input: {directory / 'daily_pv.h5'}")
        if not (self.qlib_provider / "calendars/day.txt").is_file():
            raise FileNotFoundError(f"Missing Qlib daily calendar: {self.qlib_provider}")
        for name in ("python_executable", "qlib_python"):
            executable = getattr(self, name)
            if not shutil.which(executable):
                raise FileNotFoundError(f"{name} executable not found: {executable}")
