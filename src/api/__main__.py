"""Run unified API with uvicorn: python -m src.api --port 8000."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Trustworthy AI unified API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

