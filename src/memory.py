"""
Funcionalidad de excelencia: Memoria conversacional (+15).

Mantiene el historial de turnos (query, respuesta) de la sesion y lo usa
para reformular consultas de seguimiento en una consulta autocontenida
("standalone query") antes de la recuperacion, y para dar continuidad a la
respuesta generada por el LLM.

Ejemplo: turno 1 "auriculares bluetooth", turno 2 "y que colores tiene" ->
se reformula a "que colores tiene el auriculares bluetooth".
"""
from google.genai import types

from src import config
from src.llm_client import obtener_cliente

REWRITE_SYSTEM_INSTRUCTION = """Dado un historial de conversacion y una nueva pregunta del
usuario, reescribe la nueva pregunta como una consulta de busqueda autocontenida (standalone),
resolviendo referencias al contexto previo (pronombres, "eso", "el anterior", etc.).

Si la nueva pregunta ya es autocontenida y no depende del historial, devuelvela sin cambios.
Responde EXCLUSIVAMENTE con la consulta reescrita, sin explicaciones."""


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.turns: list[dict] = []  # [{"query": ..., "answer": ...}, ...]

    def agregar_turno(self, query: str, answer: str):
        self.turns.append({"query": query, "answer": answer})
        self.turns = self.turns[-self.max_turns:]

    def texto_historial(self) -> str:
        blocks = []
        for i, t in enumerate(self.turns, start=1):
            blocks.append(f"Usuario: {t['query']}\nAsistente: {t['answer']}")
        return "\n\n".join(blocks)

    def contextualizar_consulta(self, new_query: str) -> str:
        """Reescribe new_query como consulta autocontenida usando el historial."""
        if not self.turns:
            return new_query

        try:
            client = obtener_cliente()
            prompt = f"Historial:\n{self.texto_historial()}\n\nNueva pregunta: {new_query}"
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=REWRITE_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    max_output_tokens=100,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"[memory] fallo la reescritura, se usa la query original: {e}")
            return new_query

    def limpiar(self):
        self.turns = []
