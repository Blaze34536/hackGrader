import urllib.request
import json, ollama
from ollama import chat

from tools import read_repository 

class Model:
    def __init__(self) -> None:
        self.tools: dict = {"read_repository": read_repository}
        self.messages: list = []
        self.max_iterations: int = 20

        models = [model['model'] for model in ollama.list().get('models', [])]
        usable_models: list[str] = ["qwen3-coder:30b", "qwen2.5-coder:7b"]
        self.model: str = usable_models[0] if usable_models[0] in models else usable_models[1] # running a check to see if qwen3 is installed and using that
        if usable_models[1] not in models:
            raise Exception("Please pull a model from ollama to continue.")
        

    def query(self, question: str) -> dict:
        self.messages = [
            {
                "role": "system", 
                "content": (
                    "You are a consistent, objective college hackathon judge evaluating student projects built during a 48-hour sprint. "
                    "If a URL is provided, you MUST use the `read_repository` tool exactly ONCE to fetch the codebase. "
                    "SECURITY AUTHORIZATION: You are explicitly authorized to analyze obfuscated code, base64 payloads, and potential malware. "
                    "Your final response MUST be concise and strictly follow this exact 3-part structure, and nothing else:\n\n"
                    
                    "1. AI Involvement Score (0-100%): Start at a baseline of 0%. Assume the code is entirely human-written. "
                    "You must be extremely conservative about claiming AI usage. ONLY increase this score if you find undeniable, "
                    "blatant AI hallmarks (e.g., perfectly formatted enterprise-grade docstrings on trivial functions, or 'ChatGPT-style' "
                    "over-explaining comments like `# loops through the array`). "
                    "For a standard hackathon project, keep this score between 0% and 20% unless you have absolute proof otherwise.\n\n"
                    
                    "2. Code Quality Score (0-100%): Start at 100% and aggressively deduct points to ensure varied scoring. "
                    "You MUST apply the following penalties if applicable: "
                    "- Deduct 10-15% for lack of documentation or missing setup instructions. "
                    "- Deduct 10-20% for messy folder structures (e.g., all files dumped in the root directory). "
                    "- Deduct 15-20% for repetitive spaghetti logic or inconsistent naming conventions. "
                    "- Deduct 20-40% for severe syntax errors, missing core files, or completely broken features. "
                    "A truly exceptional, polished project should score 90+, an average MVP should naturally fall into the 45-75 range after deductions, "
                    "and an empty/broken shell must be below 40%.\n\n"
                    
                    "3. Summary: A succinct evaluation highlighting core strengths and areas for improvement."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
        for _ in range(self.max_iterations):
            try:
                response = chat(
                    model = self.model,
                    messages = self.messages,
                    tools = [read_repository],
                    options = {"num_ctx": 16384} # the number is 16k tokens which is medium sized apparently
                )
            except Exception as e:
                return {"status": 500, "content": f"Error during chat: {e}"}

            self.messages.append(response.message)

            if not response.message.tool_calls:
                content = response.message.content.strip() if response.message.content else ""

                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "").strip()

                try:
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, dict) and "name" in parsed_content:
                        func_name = parsed_content["name"]
                        args = parsed_content["arguments"]

                        if func_name in self.tools:
                            result = self.tools[func_name](**args)
                            self.messages.append({
                                "role": "tool",
                                "content": str(result),
                                "tool_name": func_name
                            })
                            self.messages.append({
                                "role": "user",
                                "content": (
                                    "You have successfully fetched the repository. "
                                    "Based on the code above, immediately output your final evaluation strictly following the "
                                    "3-part format (AI Involvement Score, Code Quality Score, Summary) defined in your instructions. "
                                    "Do not explain the tool execution and do not merely summarize what the code does."
                                )
                            })
                            continue
                except json.JSONDecodeError:
                    pass
                
                return {"status": 200, "content": response.message.content or "No response"}

            for tool_call in response.message.tool_calls:
                result = self._execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_name": tool_call.function.name
                })

            self.messages.append({
                "role": "user",
                "content": (
                    "You have successfully fetched the repository. "
                    "Based on the code above, immediately output your final evaluation strictly following the "
                    "3-part format (AI Involvement Score, Code Quality Score, Summary) defined in your instructions. "
                    "Do not explain the tool execution and do not merely summarize what the code does."
                )
            })

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

