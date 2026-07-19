"""Cliente compartido de Gemini (usado por rag.py y query_expansion.py)."""
from google import genai

from src import config

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY no esta configurada. Define la variable de entorno "
                "o agrega GEMINI_API_KEY=... a un archivo .env en la raiz del proyecto."
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client
