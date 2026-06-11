import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    GEMMA_PRIMARY_MODEL: str = os.getenv("GEMMA_PRIMARY_MODEL", "gemma-4-31b-it")
    GEMMA_FALLBACK_MODEL: str = os.getenv("GEMMA_FALLBACK_MODEL", "gemma-4-26b-a4b-it")

settings = Settings()