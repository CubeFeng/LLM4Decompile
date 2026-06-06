"""FastAPI entrypoint for the LLM4Decompile demo service."""

from __future__ import annotations

from fastapi import FastAPI

from .api import router
from .logging_config import setup_logging


setup_logging()

app = FastAPI(
    title="LLM4Decompile Demo Service",
    version="0.1.0",
    description="Demo HTTP wrapper around Ghidra and LLM4Decompile vLLM inference.",
)

app.include_router(router)
