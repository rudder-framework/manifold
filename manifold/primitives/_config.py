"""Rust/Python backend toggle for all primitives."""

import os

USE_RUST = os.environ.get("MANIFOLD_USE_RUST", "1") != "0"
