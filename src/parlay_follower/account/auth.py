"""RSA-PSS request signing for the Kalshi API.

Every authenticated request carries three headers:
    KALSHI-ACCESS-KEY        -- API Key ID
    KALSHI-ACCESS-TIMESTAMP  -- current Unix time in MILLISECONDS
    KALSHI-ACCESS-SIGNATURE  -- base64(RSA-PSS-SHA256(timestamp + METHOD + path))

Gotchas (learned the hard way by everyone who integrates):
  * Sign the path WITHOUT query parameters, even if the request URL has them.
  * Timestamp must be milliseconds, not seconds.
"""
from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiSigner:
    def __init__(self, key_id: str, private_key_path: Path):
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    def headers(self, method: str, path: str) -> dict:
        """Build auth headers for one request.

        `path` must be the URL path only (e.g. '/trade-api/v2/portfolio/positions'),
        with no query string.
        """
        ts_ms = str(int(time.time() * 1000))
        message = f"{ts_ms}{method.upper()}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "Content-Type": "application/json",
        }
