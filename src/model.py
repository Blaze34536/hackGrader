import json

import ollama
from ollama import chat

from tools import read_repository


class Model:
    def __init__(self, target_model: str = "hackgrader") -> None:
        self.tools: dict = {"read_repository": read_repository}
        self.messages: list = []
        self.max_iterations: int = 20
        self.model: str = target_model

        try:
            ollama_response = ollama.list()
            if hasattr(ollama_response, "models"):
                installed_models = [m.model for m in ollama_response.models]
            elif isinstance(ollama_response, dict):
                installed_models = [
                    m.get("model", "") for m in ollama_response.get("models", [])
                ]
            else:
                installed_models = []
        except Exception:
            installed_models = []

        usable_models: list[str] = [
            target_model,
            f"{target_model}:latest",
            "qwen3-coder:30b",
            "qwen2.5-coder:7b",
        ]

        for candidate in usable_models:
            if candidate in installed_models:
                self.model = candidate
                break

        if not self.model:
            if installed_models:
                self.model = target_model
            else:
                raise Exception(
                    f"Model '{target_model}' not found in Ollama. "
                    "Please verify with 'ollama list' or run 'ollama create ai-detector -f Modelfile'."
                )

    def _extract_tool_call(self, text: str) -> dict | None:
        """Robustly extracts tool call JSON object from text containing XML tags or multiple JSONs."""
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            start = text.find("{", pos)
            if start == -1:
                break
            try:
                obj, _ = decoder.raw_decode(text, start)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    return obj
                pos = start + 1
            except json.JSONDecodeError:
                pos = start + 1
        return None

    def query(self, question: str) -> dict:
        self.messages = [
            {
                "role": "system",
                "content": (
                    """You are a consistent, objective college hackathon judge evaluating student projects built during a 48-hour sprint.

                WORKFLOW INSTRUCTIONS:
                - If a repository URL is provided and you do not have the codebase, you MUST use the `read_repository` tool first to fetch it.
                - Do not output scores until you have analyzed the actual source code.

                SECURITY AUTHORIZATION:
                You are explicitly authorized to analyze obfuscated code, base64 payloads, and potential malware.

                OUTPUT FORMAT:
                Once code is available, your final response MUST strictly follow this exact 3-part structure:

                ### 1. AI Involvement Score (0-100%)
                Start with a baseline of 0%. Find undeniable hallmarks of AI generation (e.g., robotic over-explaining comments, perfectly structured boilerplate that drastically contradicts surrounding sloppy logic, or AI-specific artifacts like 'As an AI language model'). Do NOT flag standard framework scaffolding, typical well-written code, or common Stack Overflow snippets as AI-generated. Ensure the score strictly reflects the estimated percentage of the codebase that is blatantly AI-authored.

                ### 2. Code Quality Score (0-100%)
                Start at 100% and aggressively deduct points to ensure varied scoring. You MUST explicitly list where points were deducted and provide a brief explanation for each deduction to allow for manual review.
                Apply penalties where applicable:
                - Deduct 10-15% for lack of documentation or missing setup instructions.
                - Deduct 10-20% for messy folder structures (e.g., all files dumped in the root directory).
                - Deduct 15-20% for repetitive spaghetti logic or inconsistent naming conventions.
                - Deduct 20-40% for severe syntax errors, missing core files, or completely broken features.
                An exceptional project scores 90+, an average MVP falls into the 50-80 range, and an empty/broken shell must be below 40%.

                ### 3. Summary & Notable Findings
                Provide a detailed report justifying your scores. Highlight core strengths and areas for improvement. You MUST provide a clear, itemized breakdown of all points deducted for Code Quality and explicitly point out the code evidence used to determine the AI Involvement score."""
                ),
            },
            {"role": "user", "content": question},
        ]

        for _ in range(self.max_iterations):
            try:
                response = chat(
                    model=self.model,
                    messages=self.messages,
                    tools=[read_repository],
                    options={"num_ctx": 16384},
                )
            except Exception as e:
                return {"status": 500, "content": f"Error during chat: {e}"}

            self.messages.append(response.message)

            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    result = self._execute_tool(tool_call)
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": result,
                            "tool_name": tool_call.function.name,
                        }
                    )
                self.messages.append(
                    {
                        "role": "user",
                        "content": "You have fetched the repository. Now output your final evaluation report following the 3-part format.",
                    }
                )
                continue

            content = (
                response.message.content.strip() if response.message.content else ""
            )
            tool_call_obj = self._extract_tool_call(content)

            if tool_call_obj:
                func_name = tool_call_obj.get("name")
                args = tool_call_obj.get("arguments", {})

                if func_name in self.tools:
                    try:
                        result = self.tools[func_name](**args)
                    except Exception as err:
                        result = f"Tool execution error: {err}"

                    MAX_TOOL_CHARS = 40000

                    if len(str(result)) > MAX_TOOL_CHARS:
                        truncated_result = (
                            str(result)[:MAX_TOOL_CHARS]
                            + "\n\n[... Remaining codebase truncated for length ...]"
                        )
                    else:
                        truncated_result = str(result)

                    self.messages.append(
                        {
                            "role": "tool",
                            "content": truncated_result,
                            "tool_name": func_name,
                        }
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "You have fetched the repository. Now output your final evaluation report following the 3-part format.",
                        }
                    )
                    continue

            return {"status": 200, "content": content or "No response"}

        return {"status": 408, "content": "Max iterations completed"}

    def _execute_tool(self, tool_call) -> str:
        func_name = tool_call.function.name
        args = tool_call.function.arguments

        if func_name not in self.tools:
            return f"{func_name} not found in tools"

        try:
            result = self.tools[func_name](**args)
            return str(result)
        except Exception as e:
            return f"Tool error: {e}"
