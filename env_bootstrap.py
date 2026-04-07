from pathlib import Path
from dotenv import load_dotenv


def bootstrap_env() -> Path:
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=False)

    return env_path