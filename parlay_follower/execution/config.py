"""Central configuration: settings.yaml + .env (Kalshi credentials)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional in CI
    pass

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class KalshiCreds:
    key_id: str
    private_key_path: Path
    env: str  # "demo" | "prod"


def load_settings() -> dict:
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def load_creds() -> KalshiCreds:
    key_id = os.environ.get("KALSHI_API_KEY_ID", "")
    pk_path = os.path.expanduser(os.environ.get("KALSHI_PRIVATE_KEY_PATH", ""))
    env = os.environ.get("KALSHI_ENV", "demo")
    if not key_id or not pk_path:
        raise RuntimeError(
            "Missing Kalshi credentials. Copy .env.example to .env and fill in "
            "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH."
        )
    return KalshiCreds(key_id=key_id, private_key_path=Path(pk_path), env=env)


def base_url(settings: dict, env: str) -> str:
    key = "demo_base_url" if env == "demo" else "prod_base_url"
    return settings["kalshi"][key]
