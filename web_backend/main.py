"""ASGI module for running the web backend with uvicorn."""

from __future__ import annotations

from web_backend.api import create_app

app = create_app()
