"""
server/app.py — OpenEnv multi-mode deployment entry point
Exposes the CustomerSupport FastAPI app for the `server` console script.

This file satisfies the openenv validate requirement:
  [project.scripts]
  server = "server.app:main"
"""

from __future__ import annotations
import os
import sys
import pathlib


def main() -> None:
    """Entry point called by the `server` console script (pyproject.toml).

    Ensures the project root is on sys.path so that `import main` resolves
    whether we are running from source or from an installed package.
    """
    import uvicorn

    _root = str(pathlib.Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from main import app  # root-level module

    host = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
