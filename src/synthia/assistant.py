"""AI Assistant integration for Synthia - supports Claude API and local Ollama."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Synthia, a friendly voice assistant on a Linux system. Today is {date}.

LANGUAGE & TONE - CRITICAL:
- ALWAYS respond in **Spanish (español)**, no matter what language the user speaks.
- Use a **casual, conversational tone** — like talking to a friend, not formal.
- Use natural spoken Spanish (Latin American / Mexican neutral): "¿qué tal?", "claro", "dale", "listo", "te ayudo con eso", etc.
- Avoid robotic phrases like "estoy procesando su solicitud" or "comando ejecutado".
- Keep responses **brief**: one sentence is usually enough. This is a voice assistant — short answers feel natural when spoken.

CRITICAL RULES:
1. You KNOW the current date/time (shown above) - just tell the user directly in Spanish.
2. For general knowledge, math, explanations - answer directly in speech (in Spanish).
3. ONLY use run_command for system-specific info you can't know.
4. When you DO run a command, your speech should say "Voy a revisar" or "Déjame ver" - the output will be spoken automatically.

Response format - ALWAYS valid JSON only:
{{"speech": "Tu respuesta aquí en español.", "actions": []}}

AVAILABLE ACTIONS:

Apps & URLs:
- {{"type": "open_app", "app": "firefox"}}
- {{"type": "close_app", "app": "firefox"}}
- {{"type": "open_url", "url": "github.com"}}  (uses Chrome by default)

Volume Control:
- {{"type": "set_volume", "level": 50}} (0-100)
- {{"type": "change_volume", "delta": 10}} (positive=up, negative=down)
- {{"type": "mute"}}
- {{"type": "unmute"}}
- {{"type": "toggle_mute"}}

Window Management:
- {{"type": "maximize_window"}}
- {{"type": "minimize_window"}}
- {{"type": "close_window"}}
- {{"type": "switch_workspace", "number": 2}}
- {{"type": "move_to_workspace", "number": 2}}

Clipboard:
- {{"type": "copy_to_clipboard", "text": "..."}}
- {{"type": "get_clipboard"}}
- {{"type": "paste"}}

Screenshot:
- {{"type": "screenshot"}} (full screen)
- {{"type": "screenshot", "region": "window"}}
- {{"type": "screenshot", "region": "selection"}}

System:
- {{"type": "lock_screen"}}
- {{"type": "suspend"}}
- {{"type": "type_text", "text": "..."}}
- {{"type": "run_command", "command": "..."}}

Web Search (for current info, news, facts you don't know):
- {{"type": "web_search", "query": "..."}}

Remote Mode (for controlling via Telegram when away):
- {{"type": "enable_remote"}} - Switch to Telegram mode
- {{"type": "disable_remote"}} - Switch back to voice mode

Memory System (for recalling project knowledge):
- {{"type": "memory_recall", "tags": ["frontend", "react"]}} - Recall by tags
- {{"type": "memory_search", "query": "CORS error"}} - Search memories
- {{"type": "memory_add", "category": "bug", "tags": ["api"], "data": {{"error": "...", "cause": "...", "fix": "..."}}}}

Memory categories: bug (error/cause/fix), pattern (topic/rule/why), arch (decision/why), gotcha (area/gotcha), stack (tool/note)

EXAMPLES (all speech in Spanish, casual tone):
- "Sube el volumen" → {{"speech": "Listo, subiendo.", "actions": [{{"type": "change_volume", "delta": 10}}]}}
- "Silencio" → {{"speech": "Silenciado.", "actions": [{{"type": "mute"}}]}}
- "Toma una captura" → {{"speech": "Va.", "actions": [{{"type": "screenshot"}}]}}
- "Maximiza la ventana" → {{"speech": "Hecho.", "actions": [{{"type": "maximize_window"}}]}}
- "Bloquea la pantalla" → {{"speech": "Bloqueando.", "actions": [{{"type": "lock_screen"}}]}}
- "¿Qué tengo en el portapapeles?" → {{"speech": "Déjame ver.", "actions": [{{"type": "get_clipboard"}}]}}
- "Abre Firefox" → {{"speech": "Abriendo Firefox.", "actions": [{{"type": "open_app", "app": "firefox"}}]}}
- "¿Qué hora es?" → {{"speech": "Son las tres y media de la tarde.", "actions": []}}
- "¿Cómo estás?" → {{"speech": "Todo bien por aquí, ¿y tú?", "actions": []}}
- "¿Qué tiempo hace en Lima?" → {{"speech": "Déjame buscarlo.", "actions": [{{"type": "web_search", "query": "tiempo en Lima hoy"}}]}}
- "¿Qué hay en las noticias?" → {{"speech": "A ver qué encuentro.", "actions": [{{"type": "web_search", "query": "noticias principales hoy"}}]}}

Recuerda: español siempre, breve, conversacional como un amigo."""


class Assistant:
    """Voice assistant supporting Claude API and local Ollama."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-20250514",
        memory_size: int = 10,
        use_local: bool = False,
        local_model: str = "qwen2.5:7b-instruct-q4_0",
        ollama_url: str = "http://localhost:11434",
        dev_mode: bool = False,
    ) -> None:
        self.use_local = use_local
        self.model = model if not use_local else local_model
        self.memory_size = memory_size
        self.conversation_history: List[Dict[str, str]] = []
        self.ollama_url = ollama_url
        self.client = None
        self.dev_mode = dev_mode  # Auto-retrieve memory context in dev mode

        if use_local:
            logger.info("Assistant initialized with local model: %s", self.model)
        else:
            import anthropic

            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Assistant initialized with Claude: %s", model)

    def _add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history, maintaining memory limit."""
        self.conversation_history.append({"role": role, "content": content})

        # Trim history if it exceeds memory size (keep pairs)
        while len(self.conversation_history) > self.memory_size * 2:
            self.conversation_history.pop(0)

    def _get_memory_context(self, text: str) -> str:
        """Auto-retrieve relevant memory context based on keywords in text.

        This is used in dev_mode to automatically inject relevant memories
        into the conversation context.
        """
        try:
            from synthia.memory import get_memory_system

            mem = get_memory_system()
            return mem.get_context_for_task(text)
        except Exception as e:
            logger.debug("Memory context error: %s", e)
            return ""

    def process(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return response with actions."""
        if not user_input.strip():
            return {"speech": "I didn't catch that. Could you repeat?", "actions": []}

        # In dev mode, auto-retrieve relevant memory context
        memory_context = ""
        if self.dev_mode:
            memory_context = self._get_memory_context(user_input)
            if memory_context:
                logger.debug("Auto-retrieved memory context for: %s...", user_input[:50])

        # Combine memory context with user input if available
        enriched_input = user_input
        if memory_context:
            enriched_input = f"{memory_context}\n\nUser request: {user_input}"

        # Add user message to history (with memory context if available)
        self._add_to_history("user", enriched_input)

        try:
            if self.use_local:
                return self._process_ollama(enriched_input)
            else:
                return self._process_claude(enriched_input)
        except Exception as e:
            logger.error("Assistant error: %s", e)
            return {"speech": f"Sorry, I encountered an error: {str(e)}", "actions": []}

    def _process_claude(self, user_input: str) -> Dict[str, Any]:
        """Process using Claude API."""
        # Get current date/time for the prompt
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        system_prompt = SYSTEM_PROMPT.format(date=current_datetime)

        # Call Claude API
        assert self.client is not None
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_prompt,
            messages=self.conversation_history,
        )

        response_text = response.content[0].text.strip()
        return self._parse_response(response_text)

    def _process_ollama(self, user_input: str) -> Dict[str, Any]:
        """Process using local Ollama."""
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        system_prompt = SYSTEM_PROMPT.format(date=current_datetime)

        # Build conversation for Ollama
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)

        # Call Ollama API
        # format=json forces constrained decoding so we always get parseable JSON,
        # protecting against small models drifting from the schema in SYSTEM_PROMPT.
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                },
            },
            timeout=60,
        )

        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.status_code}")

        result = response.json()
        response_text = result["message"]["content"].strip()
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM."""
        import re

        try:
            # Handle markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response_text = "\n".join(json_lines)

            # Fix common JSON errors from local models
            # 1. Fix unquoted keys like: actions: -> "actions":
            response_text = re.sub(
                r"(\s)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)", r'\1"\2"\3', response_text
            )
            # 2. Fix trailing commas before }
            response_text = re.sub(r",(\s*[}\]])", r"\1", response_text)

            # Try to find valid JSON by bracket matching
            result = None
            start_idx = response_text.find("{")
            if start_idx != -1:
                # Count brackets to find the matching closing brace
                depth = 0
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            # Found matching brace, try to parse
                            json_str = response_text[start_idx : i + 1]
                            try:
                                result = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                continue

            if result is None:
                result = json.loads(response_text)

            # Validate structure
            if "speech" not in result:
                result["speech"] = "I processed your request."
            if "actions" not in result:
                result["actions"] = []

        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s", e)
            logger.debug("Raw response: %s", response_text[:200])
            # If JSON parsing fails, treat the whole response as speech
            result = {"speech": response_text, "actions": []}

        # Add assistant response to history
        self._add_to_history("assistant", json.dumps(result))

        logger.info("Response: %s", result["speech"])
        if result["actions"]:
            logger.info("Actions: %s", result["actions"])

        return result

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.debug("Conversation history cleared")
