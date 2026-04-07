from __future__ import annotations

from env_bootstrap import bootstrap_env

bootstrap_env()

import os

import uvicorn


def main() -> None:
    host = os.getenv("LAMA_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("LAMA_SERVER_PORT", "8009"))
    uvicorn.run("tools.lama_server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
