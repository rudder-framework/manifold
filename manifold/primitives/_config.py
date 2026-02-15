"""Rust/Python backend toggle for all primitives."""

import os

USE_RUST = os.environ.get("USE_RUST", "1") != "0"
