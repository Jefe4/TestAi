# multi-agent-system/src/tools/debugging_tool.py
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool
from ..utils.jefe_datatypes import JefeContext
from ..utils.api_manager import APIManager # For type hinting

# Assuming a logger utility might be useful
try:
    from ..utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name): # type: ignore
        logger = logging.getLogger(name)
        # Basic setup if not configured by main app
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class DebuggingTool(BaseTool):
    def __init__(self, api_manager: APIManager): # APIManager is not optional for this tool
        super().__init__(
            tool_name="debug_assistant", # Changed from debugging_tool to match JefeAgent's _load_tools assumption
            tool_description="Assists in debugging code by analyzing screen content, error messages, and audio transcripts to suggest potential causes and fixes.",
            api_manager=api_manager
        )
        self.logger = get_logger(f"Tool.{self.tool_name}")

    async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
        self.logger.info(f"Executing {self.tool_name} for language: {context.programming_language or 'auto-detect'}...")
        if not self.api_manager:
            self.logger.error(f"{self.tool_name} cannot execute: APIManager not available.")
            return {"status": "error", "error": "APIManager not available for DebuggingTool.", "tool_used": self.tool_name}

        if not context.screen_content and not context.error_messages and not context.audio_transcript:
            self.logger.info(f"{self.tool_name}: No screen content, error messages, or audio transcript provided for debugging.")
            return {"status": "success",
                    "findings": "No specific code, error, or user query provided for debugging.",
                    "recommendation": "Please provide code on screen, any error messages, or describe the issue you are facing.",
                    "confidence": 0.5,
                    "tool_used": self.tool_name}

        # Limit context information to avoid excessively long prompts
        code_context = context.screen_content[:3000]
        error_context = "\n".join(context.error_messages)[:1000]
        audio_context = context.audio_transcript[:500]

        prompt = f"""
You are an expert debugging assistant. Analyze the following situation to help a developer identify and fix a problem.

Programming Language (if known): {context.programming_language or 'auto-detect'}
Current IDE (if known, for context): {context.current_ide or 'N/A'}
Project Type (if known): {context.project_type or 'N/A'}

Screen Content (Code Snippet, if relevant):
```{(context.programming_language or "").lower()}
{code_context or "No code snippet currently visible on screen."}
```

Explicit Error Messages Provided by User/IDE:
```
{error_context or "No explicit error messages were captured."}
```

User's Audio Transcript (for clues about the problem):
{audio_context or "No audio transcript provided."}

Based on all the above information, please:
1.  Identify the most likely cause(s) for the error or unexpected behavior described or implied.
2.  Suggest specific, actionable fixes or debugging steps the user can take.
3.  If the issue is unclear from the provided context, suggest what specific information the user should look for or provide next (e.g., "Check the value of variable X at line Y", "Show the full error stack trace", "What were you trying to achieve?").
4.  If multiple issues seem present, address the most critical or obvious one first.

Provide a structured response. Be empathetic and helpful.
"""
        messages = [{"role": "user", "content": prompt}]

        try:
            # This tool might benefit from a model good at reasoning and code, e.g. Gemini or a specialized Claude/Deepseek model.
            service_name = "gemini" # Default to Gemini for its reasoning capabilities
            model_name = kwargs.get("model_name", "gemini-1.5-pro-latest") # Use a more capable model if possible

            self.logger.debug(f"Calling LLM service '{service_name}' with model '{model_name}' for debugging assistance.")
            llm_response = await self.api_manager.call_llm_service(
                service_name=service_name,
                model_name=model_name,
                messages=messages,
                max_tokens=800, # Allow for detailed debugging steps
                temperature=0.4, # Balance creativity with factual debugging
                system_prompt=kwargs.get("system_prompt", "You are an expert debugging assistant AI. Your goal is to help the user identify and fix software issues.")
            )

            if llm_response.get("status") == "success" and llm_response.get("content"):
                debug_suggestions = llm_response["content"]
                self.logger.info(f"{self.tool_name} debugging suggestions generated successfully. Output length: {len(debug_suggestions)}")
                # The LLM response is expected to contain both findings and recommendations.
                return {
                    "status": "success",
                    "findings": "Debugging analysis complete. See recommendations.",
                    "recommendation": debug_suggestions, # The detailed debugging help from the LLM
                    "confidence": 0.8, # Confidence can be adjusted based on LLM's self-evaluation if available
                    "tool_used": self.tool_name
                }
            else:
                error_detail = llm_response.get("message", "LLM call for debugging failed or returned no content.")
                self.logger.error(f"{self.tool_name} failed: {error_detail}")
                return {"status": "error", "error": error_detail, "tool_used": self.tool_name}

        except Exception as e:
            self.logger.error(f"Exception during {self.tool_name} execution: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "tool_used": self.tool_name}

if __name__ == '__main__':
    import asyncio
    try:
        from ..utils.api_manager import APIManager
        from ..utils.jefe_datatypes import JefeContext
        from ..utils.logger import get_logger as setup_logger
    except ImportError: # Fallback for direct execution
        setup_logger = get_logger # Use the local one
        if 'APIManager' not in globals() or APIManager.__module__ != 'src.utils.api_manager':
            class APIManager: # type: ignore
                def __init__(self, *args, **kwargs):
                    self.logger = setup_logger("MockAPIManager_DebuggingToolTest")
                    self.service_configs: Dict[str, Any] = {}
                async def call_llm_service(self, service_name: str, model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.3, **kwargs: Any) -> Dict[str, Any]:
                    self.logger.info(f"MockAPIManager: call_llm_service for DebuggingToolTest with {service_name} model {model_name}")
                    prompt_content = messages[0]['content']
                    if "ZeroDivisionError" in prompt_content:
                        return {"status": "success", "content": "Suggestion: Check for division by zero. Ensure denominators are not zero before performing division."}
                    return {"status": "success", "content": f"Mock Debugging Suggestions for context: {prompt_content[:150]}..."}

        if 'JefeContext' not in globals() or JefeContext.__module__ != 'src.utils.jefe_datatypes':
            @dataclass
            class JefeContext: # type: ignore
                screen_content: str
                audio_transcript: str
                current_ide: Optional[str] = None
                programming_language: Optional[str] = None
                project_type: Optional[str] = None
                error_messages: List[str] = field(default_factory=list)
                previous_suggestions: List[str] = field(default_factory=list)
                def summarize(self, max_screen_len: int = 200, max_audio_len: int = 100) -> str: return "Mock context summary for debug tool test"


    async def test_debugging_tool():
        print("--- Testing DebuggingTool ---")

        mock_api_manager = APIManager() # Uses fallback if real one not imported
        if not hasattr(mock_api_manager, 'service_configs'): mock_api_manager.service_configs = {}
        if "gemini" not in mock_api_manager.service_configs: # Tool defaults to gemini
             mock_api_manager.service_configs["gemini"] = {"api_key":"dummy_key_for_debug_tool", "base_url":"http://localhost/dummy_gemini"}

        tool = DebuggingTool(api_manager=mock_api_manager)
        print(f"Tool Description: {tool.get_description()}")

        test_context_error = JefeContext(
            screen_content="x = 10\ny = x / 0 # Potential error here",
            audio_transcript="User: My Python script is crashing with an error when I run it.",
            programming_language="Python",
            error_messages=["Traceback (most recent call last):\n  File \"test.py\", line 2, in <module>\n    y = x / 0\nZeroDivisionError: division by zero"]
        )
        result_error = await tool.execute(test_context_error)
        print(f"\nError Debugging Result:\n{result_error}")
        assert result_error["status"] == "success"
        assert "division by zero" in result_error.get("recommendation", "").lower()

        test_context_no_specific_error = JefeContext(
            screen_content="for i in range(5):\n  print(i)\n  # Code seems fine but user is confused",
            audio_transcript="User: This loop isn't doing what I expect, it stops too early.",
            programming_language="Python"
        )
        result_no_error = await tool.execute(test_context_no_specific_error)
        print(f"\nNo Explicit Error Debugging Result:\n{result_no_error}")
        assert result_no_error["status"] == "success"
        assert "Mock Debugging Suggestions" in result_no_error.get("recommendation", "")

        test_context_no_info = JefeContext(screen_content="", audio_transcript="", error_messages=[])
        result_no_info = await tool.execute(test_context_no_info)
        print(f"\nNo Information Provided Result:\n{result_no_info}")
        assert result_no_info.get("findings") == "No specific code, error, or user query provided for debugging."

        print("\n--- DebuggingTool testing finished ---")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_debugging_tool())
