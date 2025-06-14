# multi-agent-system/src/tools/architecture_tool.py
from typing import Dict, Any, Optional, List # Added List

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

class ArchitectureAdvisorTool(BaseTool):
    def __init__(self, api_manager: APIManager):
        super().__init__(
            tool_name="architecture_advisor",
            tool_description="Provides architectural guidance, suggests design patterns, and advises on scalability and maintainability based on the current code context and user queries.",
            api_manager=api_manager
        )
        self.logger = get_logger(f"Tool.{self.tool_name}")

    async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
        self.logger.info(f"Executing {self.tool_name}...")
        if not self.api_manager:
            return {"status": "error", "error": "APIManager not available.", "tool_used": self.tool_name, "confidence": 0.0}

        if not context.screen_content and not context.audio_transcript and not context.general_query:
            return {
                "status": "success", # Or "info"
                "primary_finding": "No specific code or architectural question provided for advice.", # Changed key to match synthesis expectations
                "recommendation": "Please provide code context, use the general query field, or describe your architectural challenge via audio.",
                "confidence": 0.3, # Low confidence as no action taken
                "tool_used": self.tool_name
            }

        code_context = context.screen_content[:3000] # Limit length
        audio_context = context.audio_transcript[:1000] # Limit length
        general_query = context.general_query[:1000] if context.general_query else ""


        prompt = f"""
You are an expert software architect. Based on the following context, provide architectural guidance.

Programming Language (if known): {context.programming_language or 'N/A'}
Project Type (if known): {context.project_type or 'N/A'}
Current IDE (if known): {context.current_ide or 'N/A'}
General Query from User: {general_query or "No specific general query provided."}

Screen Content (Code Snippet, if relevant):
```
{code_context or "No specific code snippet provided on screen."}
```

User's Audio Transcript (for goals, questions, or problems):
{audio_context or "No specific audio input."}

Please consider the following aspects in your advice:
1.  Suitable design patterns or architectural styles.
2.  Potential scalability issues and solutions.
3.  Maintainability and code organization best practices.
4.  Technology stack appropriateness (if inferable or mentioned).
5.  Integration patterns with other services or components.
6.  Trade-offs of different approaches.

Provide clear, actionable recommendations and explain your reasoning.
Structure your response. Aim for these sections if possible:
- "Primary Issue/Goal": Briefly state the core problem or architectural goal.
- "Recommended Patterns/Approaches": Suggest specific patterns or architectural styles.
- "Key Considerations": Discuss important factors like scalability, maintainability, trade-offs.
- "Implementation Tips": Offer actionable advice for implementation.
"""
        messages = [{"role": "user", "content": prompt}]

        try:
            # Use a capable model for architectural advice, e.g., Claude Opus or Gemini Pro
            # Allow model to be passed via kwargs or use a default from config if available
            service_name = kwargs.get("service_name", "claude") # Default to Claude for potentially longer, nuanced responses
            model_name = kwargs.get("model_name", "claude-3-opus-20240229") # Or a strong Gemini model

            llm_response = await self.api_manager.call_llm_service(
                service_name=service_name,
                model_name=model_name,
                messages=messages,
                max_tokens=1200, # Architectural advice can be lengthy
                temperature=0.5 # Balance creativity and precision
            )

            if llm_response.get("status") == "success" and llm_response.get("content"):
                architectural_advice = llm_response["content"]
                self.logger.info(f"{self.tool_name} advice generation successful.")
                # The synthesis logic in JefeAgent will extract primary_issue, recommended_action, etc.
                # This tool should return the core advice. JefeAgent's synthesis can parse it.
                # For direct use, we can try to provide some basic parsing or rely on the LLM's structure.

                # Basic parsing attempt (example - could be more sophisticated)
                # This is a simple heuristic. A more robust approach might involve specific LLM prompting for JSON.
                parsed_advice = {"raw_text": architectural_advice}
                lines = architectural_advice.splitlines()
                current_section = None
                for line in lines:
                    if line.startswith("Primary Issue/Goal:"):
                        current_section = "primary_finding" # Match JefeAgent synthesis key
                        parsed_advice[current_section] = line.split(":", 1)[1].strip()
                    elif line.startswith("Recommended Patterns/Approaches:"):
                        current_section = "recommendation" # Match JefeAgent synthesis key
                        parsed_advice[current_section] = line.split(":", 1)[1].strip()
                    elif line.startswith("Key Considerations:"):
                        current_section = "key_considerations"
                        parsed_advice[current_section] = line.split(":", 1)[1].strip()
                    elif line.startswith("Implementation Tips:"):
                        current_section = "additional_value" # Match JefeAgent synthesis key for enhancement_tip
                        parsed_advice[current_section] = line.split(":", 1)[1].strip()
                    elif current_section and line.strip() and not line.strip().startswith("- "): # append to current section if it's a continuation
                        if parsed_advice.get(current_section) and isinstance(parsed_advice[current_section], str):
                             parsed_advice[current_section] += "\n" + line.strip()


                return {
                    "status": "success",
                    "data": { # Nesting advice under 'data' as per other tools
                        "architectural_advice_raw": architectural_advice,
                        "primary_finding": parsed_advice.get("primary_finding", "See raw architectural advice for details."),
                        "recommendation": parsed_advice.get("recommendation", "See raw architectural advice for details."),
                        "key_considerations": parsed_advice.get("key_considerations", ""),
                        "additional_value": parsed_advice.get("additional_value", "Further details in raw advice.") # For enhancement_tip
                    },
                    "confidence": 0.8,
                    "tool_used": self.tool_name
                }
            else:
                error_detail = llm_response.get("message", "LLM call failed or returned no content for architecture tool.")
                self.logger.error(f"{self.tool_name} failed: {error_detail}")
                return {"status": "error", "error": error_detail, "tool_used": self.tool_name, "confidence": 0.1}

        except Exception as e:
            self.logger.error(f"Exception in {self.tool_name}: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "tool_used": self.tool_name, "confidence": 0.1}

if __name__ == '__main__':
    import asyncio
    # Adjust imports for direct execution if APIManager/JefeContext are in different relative paths
    import sys
    import os
    # Assuming the script is in src/tools/ and we need to go up to src/ then to utils
    project_root_for_test = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)

    from src.utils.api_manager import APIManager # For main test
    from src.utils.jefe_datatypes import JefeContext # For main test
    from src.utils.logger import get_logger as get_root_logger # To avoid conflict if logger is top-level in this file

    # Re-configure logger for the test setup if it was basic before
    logger_main_test = get_root_logger("ArchitectureToolTest")


    class MockAPIManager(APIManager):
        def __init__(self, service_configs: Optional[Dict[str, Dict[str, str]]] = None):
            # Pass a basic config, but it won't be used for API calls by the mock
            super().__init__(service_configs if service_configs else {"claude": {"api_key": "dummy", "base_url": "dummy"}})
            self.logger = get_root_logger("MockAPIManager.ArchitectureToolTest")


        async def call_llm_service(self, service_name: str, model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.3, **kwargs: Any) -> Dict[str, Any]:
            self.logger.info(f"MockAPIManager: call_llm_service for ArchitectureAdvisorTool with {service_name} model {model_name}")
            prompt_content = messages[0]['content']
            # Simulate a structured-like response
            return {
                "status": "success",
                "content": f"Mock Architectural Advice for: {prompt_content[:150]}...\nPrimary Issue/Goal: Scalability concern for user services.\nRecommended Patterns/Approaches: Microservices, CQRS, Event Sourcing.\nKey Considerations: Data consistency, eventual consistency trade-offs, inter-service communication (e.g., gRPC, message queues).\nImplementation Tips: Start with a well-defined bounded context for the OrderService. Use asynchronous communication for payment processing to improve resilience."
            }

    async def test_architecture_advisor_tool():
        logger_main_test.info("--- Testing ArchitectureAdvisorTool ---")
        # Provide a minimal config for APIManager initialization
        mock_api_manager = MockAPIManager(service_configs={"claude": {"api_key": "dummy_key", "base_url": "http://localhost:8000"}})

        tool = ArchitectureAdvisorTool(api_manager=mock_api_manager)
        logger_main_test.info(tool.get_description())

        test_context_arch = JefeContext(
            screen_content="class OrderService { processOrder() { /* ... */ } }\nclass PaymentService { charge() { /* ... */ } }",
            audio_transcript="I'm designing a new e-commerce backend and need advice on how to structure the order and payment services. Should they be microservices?",
            programming_language="Java",
            project_type="E-commerce Backend",
            general_query="What are the best practices for decoupling these services?"
        )
        result_arch = await tool.execute(test_context_arch, service_name="claude", model_name="claude-3-opus-20240229")
        logger_main_test.info(f"Architecture Advice Result: {result_arch}")
        assert result_arch["status"] == "success"
        assert "Mock Architectural Advice" in result_arch.get("data", {}).get("architectural_advice_raw", "")
        assert "Scalability concern" in result_arch.get("data", {}).get("primary_finding", "")
        assert "Microservices" in result_arch.get("data", {}).get("recommendation", "")


        test_context_no_info = JefeContext(screen_content="", audio_transcript="", general_query="")
        result_no_info = await tool.execute(test_context_no_info)
        logger_main_test.info(f"No Info Result: {result_no_info}")
        assert result_no_info.get("primary_finding") == "No specific code or architectural question provided for advice." # Check updated key

    if os.name == 'nt': # Required for Windows if using asyncio.run in some contexts
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_architecture_advisor_tool())
