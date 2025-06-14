# multi-agent-system/src/tools/code_analysis_tool.py
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

class CodeAnalysisTool(BaseTool):
    def __init__(self, api_manager: APIManager): # APIManager is not optional for this tool
        super().__init__(
            tool_name="code_analyzer", # Changed from code_analyzer_tool to match JefeAgent's _load_tools assumption
            tool_description="Analyzes provided code snippets for syntax errors, logic issues, readability, and best practice adherence. Returns findings and suggestions.",
            api_manager=api_manager
        )
        self.logger = get_logger(f"Tool.{self.tool_name}")

    async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
        self.logger.info(f"Executing {self.tool_name} for language: {context.programming_language or 'auto-detect'}...")
        if not self.api_manager:
            self.logger.error(f"{self.tool_name} cannot execute: APIManager not available.")
            return {"status": "error", "error": "APIManager not available for CodeAnalysisTool.", "tool_used": self.tool_name}

        if not context.screen_content:
            self.logger.info(f"{self.tool_name}: No screen content provided for analysis.")
            return {"status": "success", # Or "no_op"? For now, success as tool executed.
                    "findings": "No code provided in screen content for analysis.",
                    "recommendation": "Ensure code is visible on screen or provide it directly.",
                    "confidence": 0.5,
                    "tool_used": self.tool_name}

        # Limit screen content to avoid excessive prompt length
        code_to_analyze = context.screen_content[:3000] # Approx 3k chars limit for the code part

        prompt = f"""
Analyze the following code snippet.
Focus on identifying:
1. Syntax errors and critical typos.
2. Potential logic errors, bugs, or anti-patterns.
3. Readability issues and deviations from common code style conventions for the language.
4. Missing error handling or unaddressed edge cases.
5. Basic security vulnerabilities (e.g., SQL injection if SQL-like, XSS if web code - be cautious).
6. Concrete suggestions for improvement, refactoring, or alternative approaches.

Programming Language (if known, otherwise try to infer): {context.programming_language or 'auto-detect'}
Current IDE (if known, for context): {context.current_ide or 'N/A'}
Relevant error messages from context (if any): {' ; '.join(context.error_messages) if context.error_messages else 'None'}

Code to Analyze:
```{(context.programming_language or "").lower()}
{code_to_analyze}
```

Provide your analysis as a structured response. Highlight key findings and offer actionable recommendations.
If the code appears generally sound, state so and offer minor suggestions for style or clarity if applicable.
Be concise and direct in your findings and recommendations.
"""
        messages = [{"role": "user", "content": prompt}]

        try:
            # Using Gemini as an example, but this could be configurable or use a specific code analysis model.
            service_name = "gemini"
            model_name = kwargs.get("model_name", "gemini-1.5-flash-latest") # Allow override via kwargs

            self.logger.debug(f"Calling LLM service '{service_name}' with model '{model_name}' for code analysis.")
            llm_response = await self.api_manager.call_llm_service(
                service_name=service_name,
                model_name=model_name,
                messages=messages,
                max_tokens=800, # Allow for a reasonably detailed analysis
                temperature=0.2, # Low temperature for factual, analytical output
                system_prompt=kwargs.get("system_prompt", "You are an expert code reviewer and static analysis tool.") # Optional system prompt for the LLM call
            )

            if llm_response.get("status") == "success" and llm_response.get("content"):
                analysis_text = llm_response["content"]
                self.logger.info(f"{self.tool_name} analysis successful. Output length: {len(analysis_text)}")
                # The LLM is expected to provide findings and recommendations within its content.
                # We could try to parse this further if the LLM provides structured output (e.g., JSON mode).
                return {
                    "status": "success",
                    "findings": analysis_text, # The full analysis from the LLM
                    "recommendation": "Review the detailed analysis provided in 'findings'. Implement suggested changes as appropriate.",
                    "confidence": 0.85, # Confidence in the LLM's ability to analyze, can be adjusted
                    "tool_used": self.tool_name
                }
            else:
                error_detail = llm_response.get("message", "LLM call for code analysis failed or returned no content.")
                self.logger.error(f"{self.tool_name} failed: {error_detail}")
                return {"status": "error", "error": error_detail, "tool_used": self.tool_name}

        except Exception as e:
            self.logger.error(f"Exception during {self.tool_name} execution: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "tool_used": self.tool_name}

if __name__ == '__main__':
    import asyncio
    # Need to ensure APIManager and JefeContext are correctly importable for this test
    # This might require adjusting sys.path or running from the project root.
    try:
        from ..utils.api_manager import APIManager
        from ..utils.jefe_datatypes import JefeContext
        from ..utils.logger import get_logger as setup_logger # To avoid conflict with local get_logger
    except ImportError: # Fallback for direct execution
        setup_logger = get_logger # Use the local one
        # Define Fallback APIManager and JefeContext if needed for standalone test
        if 'APIManager' not in globals() or APIManager.__module__ != 'src.utils.api_manager':
            class APIManager: # type: ignore
                def __init__(self, *args, **kwargs):
                    self.logger = setup_logger("MockAPIManager_CAToolTest")
                    self.service_configs: Dict[str, Any] = {}
                async def call_llm_service(self, service_name: str, model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.3, **kwargs: Any) -> Dict[str, Any]:
                    self.logger.info(f"MockAPIManager: call_llm_service for CodeAnalysisToolTest with {service_name} model {model_name}")
                    prompt_content = messages[0]['content']
                    # Simulate different responses based on prompt for testing
                    if "syntax error" in prompt_content.lower():
                        return {"status": "success", "content": "Syntax Analysis: Found a syntax error. Recommendation: Fix the colon."}
                    return {"status": "success", "content": f"Mock LLM Code Analysis for: {prompt_content[:150]}..."}
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
                def summarize(self, max_screen_len: int = 200, max_audio_len: int = 100) -> str: return "Mock context summary"


    async def test_code_analysis_tool():
        print("--- Testing CodeAnalysisTool ---")

        mock_api_manager = APIManager() # Uses fallback if real one not imported
        # Ensure the mock_api_manager has the 'gemini' service configured for the tool's default
        if not hasattr(mock_api_manager, 'service_configs'): mock_api_manager.service_configs = {}
        if "gemini" not in mock_api_manager.service_configs:
             mock_api_manager.service_configs["gemini"] = {"api_key":"dummy_key_for_cat_test", "base_url":"http://localhost/dummy_gemini"}

        tool = CodeAnalysisTool(api_manager=mock_api_manager)
        print(f"Tool Description: {tool.get_description()}")

        test_context_py = JefeContext(
            screen_content="def my_func(a,b)\n  return a+b", # Intentional Python syntax error
            audio_transcript="User: Can you review this Python code snippet for me?",
            programming_language="Python",
            error_messages=[]
        )
        result_py = await tool.execute(test_context_py)
        print(f"\nPython Code Analysis Result:\n{result_py}")
        assert result_py["status"] == "success"
        assert "Mock LLM Code Analysis" in result_py.get("findings", "") # Or specific error if mock handles it

        test_context_js = JefeContext(
            screen_content="function greet(name) { console.log('Hello ' + name); }",
            audio_transcript="User: What do you think of this JavaScript function?",
            programming_language="JavaScript"
        )
        result_js = await tool.execute(test_context_js)
        print(f"\nJavaScript Code Analysis Result:\n{result_js}")
        assert result_js["status"] == "success"

        test_context_no_code = JefeContext(screen_content="", audio_transcript="User is silent, nothing on screen.")
        result_no_code = await tool.execute(test_context_no_code)
        print(f"\nNo Code Provided Result:\n{result_no_code}")
        assert result_no_code.get("findings") == "No code provided in screen content for analysis."

        print("\n--- CodeAnalysisTool testing finished ---")

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_code_analysis_tool())
