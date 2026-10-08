import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    def __init__(self):
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.PROJECTS_DIR: Path = Path(os.getenv("PROJECTS_DIR", str(BASE_DIR / "projects")))
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")

        # Ensure projects directory exists
        self.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
