# multi-agent-system/src/tools/performance_tool.py
from typing import Dict, Any, Optional, List

from .base_tool import BaseTool
from ..utils.jefe_datatypes import JefeContext
from ..utils.api_manager import APIManager # For type hinting

# Assuming a logger utility
try:
    from ..utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name): # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class PerformanceOptimizerTool(BaseTool):
    def __init__(self, api_manager: APIManager):
        super().__init__(
            tool_name="performance_optimizer",
            tool_description="Analyzes code and context for performance bottlenecks and suggests optimizations. Considers algorithms, data structures, I/O operations, and concurrency.",
            api_manager=api_manager
        )
        self.logger = get_logger(f"Tool.{self.tool_name}")

    async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
        self.logger.info(f"Executing {self.tool_name}...")
        if not self.api_manager:
            return {"status": "error", "error": "APIManager not available.", "tool_used": self.tool_name, "confidence": 0.0}

        # Check if there's enough specific context related to performance
        if not context.screen_content and \
           not context.audio_transcript and \
           not context.error_messages and \
           not context.general_query: # Added general_query check
            return {
                "status": "success", # Or "info"
                "primary_finding": "No specific code, performance issue description, or error messages provided for optimization advice.", # Key changed for JefeAgent
                "recommendation": "Please provide code, describe the performance problem (via audio or general query), or share relevant errors (e.g., timeouts).",
                "confidence": 0.3, # Low confidence as no action taken
                "tool_used": self.tool_name
            }

        code_context = context.screen_content[:3500] # Slightly increased limit
        audio_context = context.audio_transcript[:1500] # Slightly increased limit
        error_context = "\n".join(context.error_messages)[:1500] # Slightly increased limit
        general_query = context.general_query[:1000] if context.general_query else ""


        prompt = f"""
You are an expert software performance engineer. Analyze the provided context for performance issues and suggest optimizations.

Programming Language (if known): {context.programming_language or 'N/A'}
Project Type (if known): {context.project_type or 'N/A'}
General Query from User: {general_query or "No specific general query related to performance."}

Screen Content (Code Snippet, if relevant):
```
{code_context or "No specific code snippet provided on screen."}
```

User's Audio Transcript (for description of slowness, goals, or problems):
```
{audio_context or "No specific audio input describing performance issues."}
```

Error Messages Provided (e.g., timeouts, out-of-memory):
```
{error_context or "No explicit error messages suggesting performance issues."}
```

Based on this information, please:
1.  Identify potential performance bottlenecks (e.g., inefficient algorithms, slow I/O, excessive computation, memory leaks, inefficient data structures, concurrency issues).
2.  Suggest specific, actionable optimizations or code refactoring. Be precise.
3.  If applicable, recommend profiling tools or techniques for the given language/platform to further diagnose the issue.
4.  Explain the trade-offs of your suggested optimizations (e.g., complexity, memory usage).
5.  Structure your response clearly. If possible, use sections like:
    - "Identified Bottleneck(s)":
    - "Optimization Suggestions":
    - "Profiling Recommendations":
    - "Trade-offs":

If no clear performance issues are evident from the provided context, suggest general performance best practices relevant to the language/project type if possible.
"""
        messages = [{"role": "user", "content": prompt}]

        try:
            # Use a model adept at code analysis and performance understanding
            service_name = kwargs.get("service_name", "gemini") # Default to Gemini
            model_name = kwargs.get("model_name", "gemini-1.5-pro-latest") # A strong model

            llm_response = await self.api_manager.call_llm_service(
                service_name=service_name,
                model_name=model_name,
                messages=messages,
                max_tokens=1000, # Performance advice can be detailed
                temperature=0.25 # More factual, less creative for performance
            )

            if llm_response.get("status") == "success" and llm_response.get("content"):
                performance_advice = llm_response["content"]
                self.logger.info(f"{self.tool_name} analysis successful.")

                # Basic parsing attempt for JefeAgent synthesis (similar to ArchitectureTool)
                parsed_advice = {"raw_text": performance_advice}
                lines = performance_advice.splitlines()
                current_section = None
                # Heuristic keys to match common LLM outputs for performance
                section_map = {
                    "Identified Bottleneck(s)": "primary_finding", # For JefeAgent's primary_issue
                    "Optimization Suggestions": "recommendation",    # For JefeAgent's recommended_action
                    "Profiling Recommendations": "profiling_tips",
                    "Trade-offs": "trade_offs",
                    # General "Summary" or "Conclusion" could be additional_value
                    "Summary": "additional_value",
                    "Conclusion": "additional_value"
                }

                for line in lines:
                    matched_section_key = None
                    for header, key_in_parsed_advice in section_map.items():
                        if line.startswith(header + ":"):
                            current_section = key_in_parsed_advice
                            parsed_advice[current_section] = line.split(":",1)[1].strip()
                            matched_section_key = True
                            break
                        elif line.startswith("- " + header + ":"): # Handle markdown list like headers
                            current_section = key_in_parsed_advice
                            parsed_advice[current_section] = line.split(":",1)[1].strip()
                            matched_section_key = True
                            break
                    if matched_section_key: continue

                    if current_section and line.strip() and not line.strip().startswith("- "): # append to current section
                         if parsed_advice.get(current_section) and isinstance(parsed_advice[current_section], str):
                             parsed_advice[current_section] += "\n" + line.strip()

                # Ensure primary_finding and recommendation have fallbacks for JefeAgent
                primary_finding_val = parsed_advice.get("primary_finding", "Detailed analysis in raw text. No single bottleneck summarized by tool.")
                recommendation_val = parsed_advice.get("recommendation", "See raw performance analysis for optimization suggestions.")


                return {
                    "status": "success",
                    "data": { # Nesting advice under 'data'
                        "performance_analysis_raw": performance_advice,
                        "primary_finding": primary_finding_val,
                        "recommendation": recommendation_val,
                        "profiling_tips": parsed_advice.get("profiling_tips", ""),
                        "trade_offs": parsed_advice.get("trade_offs", ""),
                        "additional_value": parsed_advice.get("additional_value", "") # For enhancement_tip
                    },
                    "confidence": 0.75, # Confidence in the LLM's analysis (adjust based on model)
                    "tool_used": self.tool_name
                }
            else:
                error_detail = llm_response.get("message", "LLM call failed or returned no content for performance tool.")
                self.logger.error(f"{self.tool_name} failed: {error_detail}")
                return {"status": "error", "error": error_detail, "tool_used": self.tool_name, "confidence": 0.1}

        except Exception as e:
            self.logger.error(f"Exception in {self.tool_name}: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "tool_used": self.tool_name, "confidence": 0.1}

if __name__ == '__main__':
    import asyncio
    # Adjust imports for direct execution
    import sys
    import os
    project_root_for_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)

    from src.utils.api_manager import APIManager # For main test
    from src.utils.jefe_datatypes import JefeContext # For main test
    from src.utils.logger import get_logger as get_root_logger # To avoid conflict

    logger_main_test = get_root_logger("PerformanceOptimizerToolTest")

    class MockAPIManager(APIManager):
        def __init__(self, service_configs: Optional[Dict[str, Dict[str, str]]] = None):
            super().__init__(service_configs if service_configs else {"gemini": {"api_key": "dummy", "base_url": "dummy"}})
            self.logger = get_root_logger("MockAPIManager.PerformanceOptimizerToolTest")

        async def call_llm_service(self, service_name: str, model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.3, **kwargs: Any) -> Dict[str, Any]:
            self.logger.info(f"MockAPIManager: call_llm_service for PerformanceOptimizerTool with {service_name} model {model_name}")
            prompt_content = messages[0]['content']
            # Simulate a structured-like response
            return {
                "status": "success",
                "content": (
                    f"Mock Performance Analysis for: {prompt_content[:100]}...\n"
                    "Identified Bottleneck(s): The primary bottleneck appears to be the nested loop structure causing O(n^2) complexity.\n"
                    "Optimization Suggestions: Consider vectorizing the operation if using libraries like NumPy/Pandas. If not, try to reduce loop depth or use a more efficient algorithm for the task.\n"
                    "Profiling Recommendations: Use cProfile for Python to identify exact hotspots.\n"
                    "Trade-offs: Vectorization might increase memory temporarily. Algorithmic changes might require more complex code."
                )
            }

    async def test_performance_optimizer_tool():
        logger_main_test.info("--- Testing PerformanceOptimizerTool ---")
        mock_api_manager = MockAPIManager()

        tool = PerformanceOptimizerTool(api_manager=mock_api_manager)
        logger_main_test.info(tool.get_description())

        test_context_perf = JefeContext(
            screen_content="for i in range(10000):\n  for j in range(10000):\n    # do something small",
            audio_transcript="This Python code with nested loops is extremely slow. It's taking ages to complete.",
            programming_language="Python",
            project_type="Data Processing Script",
            general_query="How can I make these nested loops faster?"
        )
        result_perf = await tool.execute(test_context_perf) # Uses default gemini
        logger_main_test.info(f"Performance Analysis Result: {result_perf}")
        assert result_perf["status"] == "success"
        raw_advice = result_perf.get("data", {}).get("performance_analysis_raw", "")
        assert "Mock Performance Analysis" in raw_advice
        assert "nested loop structure" in result_perf.get("data", {}).get("primary_finding", "")
        assert "vectorizing the operation" in result_perf.get("data", {}).get("recommendation", "")


        test_context_no_info = JefeContext(screen_content="", audio_transcript="", error_messages=[], general_query="")
        result_no_info = await tool.execute(test_context_no_info)
        logger_main_test.info(f"No Info Result: {result_no_info}")
        assert result_no_info.get("primary_finding") == "No specific code, performance issue description, or error messages provided for optimization advice." # Check updated key

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_performance_optimizer_tool())
