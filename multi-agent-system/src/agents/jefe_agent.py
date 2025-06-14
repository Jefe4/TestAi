# multi-agent-system/src/agents/jefe_agent.py
from typing import Dict, Any, Optional, List
import asyncio
import importlib # For dynamic tool loading
import os # For dynamic tool loading

from .base_agent import BaseAgent
from ..utils.api_manager import APIManager
from ..utils.jefe_datatypes import JefeContext, JefeTaskType
from ..utils.jefe_task_analyzer import JefeTaskAnalyzer, TaskAnalysis
from ..utils.jefe_context_manager import JefeContextManager
# Use TYPE_CHECKING for BaseTool to avoid definite circular import issues at runtime
# if tools themselves might import something from agents or utils that import agents.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..tools.base_tool import BaseTool


# Assuming a logger utility exists
try:
    from ..utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name): # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers: # Basic setup if no handlers configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class JefeAgent(BaseAgent):
    """
    JefeAgent: A specialized AI assistant for real-time coding and development support.

    This agent analyzes the user's context (screen, audio, IDE info),
    determines the task at hand, and utilizes a suite of tools (potentially in parallel)
    to provide assistance, suggestions, or perform actions. It aims for a proactive,
    single-agent-multiple-tools operational pattern.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the JefeAgent.

        Args:
            agent_name: The name for this instance of JefeAgent.
            api_manager: An instance of APIManager for external API calls (used by context manager and tools).
            config: Optional configuration dictionary for the agent.
        """
        super().__init__(agent_name, config if config else {})
        self.api_manager = api_manager
        # Ensure logger uses the specific agent name for better log filtering
        self.logger = get_logger(f"JefeAgent.{agent_name}")

        # Jefe-specific components
        self.task_analyzer = JefeTaskAnalyzer()
        # Pass api_manager to JefeContextManager for its context compression capabilities
        self.context_manager = JefeContextManager(api_manager=self.api_manager)

        # Dynamically load available tools from the tools directory
        self.tools: Dict[str, 'BaseTool'] = self._load_tools()
        self.logger.info(f"JefeAgent '{agent_name}' initialized. Loaded {len(self.tools)} tools: {list(self.tools.keys()) if self.tools else 'None'}")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the JefeAgent.
        """
        return {
            "description": "Jefe: A real-time coding assistant with screen/audio analysis and tool usage. Aims for proactive, context-aware support using a single-agent, parallel-tools approach.",
            "capabilities": [
                "realtime_context_analysis", "proactive_coding_assistance",
                "debugging_support", "architecture_advice",
                "performance_tuning_tips", "contextual_documentation_lookup",
                "code_generation_and_refactoring"
            ],
            "supported_task_types": [str(t.value) for t in JefeTaskType], # List all task types it can identify
            "tools_available": list(self.tools.keys()) # List names of loaded tools
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard BaseAgent interface adapter.

        For JefeAgent, most interactions are expected to come via `process_realtime_input`
        with a full `JefeContext`. This method handles simpler text-based queries
        by adapting them into a minimal `JefeContext`.

        Args:
            query_data: A dictionary, typically expecting a "prompt" key for text input.
                        May also contain "current_ide", "programming_language", "project_type".

        Returns:
            A dictionary containing the agent's response, formatted by `_format_jefe_response`.
        """
        prompt = query_data.get("prompt", "")
        self.logger.info(f"Processing standard text query via JefeAgent: '{prompt[:100]}...'")

        # Create a minimal JefeContext for this text-based query.
        # The prompt is treated as the primary "screen content".
        context = JefeContext(
            screen_content=prompt,
            audio_transcript="", # No audio transcript for a simple text query
            current_ide=query_data.get("current_ide"),
            programming_language=query_data.get("programming_language"),
            project_type=query_data.get("project_type")
            # error_messages and previous_suggestions will be empty by default
        )
        # Delegate to the main real-time input processing logic
        return await self.process_realtime_input(context)

    async def process_realtime_input(self, jefe_context: JefeContext) -> Dict[str, Any]:
        """
        Main processing method for JefeAgent, designed for real-time contextual input.

        This method orchestrates the agent's response by:
        1. Logging the incoming context.
        2. Analyzing the context using `JefeTaskAnalyzer`.
        3. Determining an appropriate handling strategy based on task complexity.
        4. Executing the strategy (simple assistance, parallel analysis, or complex research).
        5. Formatting the raw result into a user-facing response.
        6. Logging the final response as part of the interaction history.

        Args:
            jefe_context: A `JefeContext` object containing screen content, audio transcript,
                          IDE information, errors, etc.

        Returns:
            A dictionary representing the agent's formatted response to the input context.
        """
        self.logger.info(f"Processing real-time input. Screen: {len(jefe_context.screen_content)} chars, Audio: {len(jefe_context.audio_transcript)} chars, Errors: {len(jefe_context.error_messages)}.")
        # Add current context to history (response will be added later)
        # Storing the context object directly. Response will be the raw_result before formatting for this interaction.
        # self.context_manager.add_interaction(jefe_context, {"status": "processing_input"})

        # 1. Analyze the context
        task_analysis_result: TaskAnalysis = self.task_analyzer.analyze_context(jefe_context)
        self.logger.info(f"JefeTaskAnalyzer analysis: Type='{task_analysis_result.task_type.value}', Complexity='{task_analysis_result.complexity}', Priority='{task_analysis_result.priority}', Issues='{task_analysis_result.immediate_issues}', Tools='{task_analysis_result.suggested_tools}'")

        # Add analysis to trace (optional, could also be part of the final response log)
        # self.context_manager.add_trace_event("jefe_task_analysis_complete", TaskAnalysis.to_dict(task_analysis_result)) # if TaskAnalysis has to_dict

        raw_result: Dict[str, Any]
        # 2. Determine handling strategy based on complexity
        if task_analysis_result.complexity == "simple":
            raw_result = await self._handle_simple_assistance(jefe_context, task_analysis_result)
        elif task_analysis_result.complexity == "moderate":
            raw_result = await self._handle_parallel_analysis(jefe_context, task_analysis_result)
        else: # "complex"
            raw_result = await self._handle_complex_research(jefe_context, task_analysis_result)

        # 3. Format the raw result into a user-facing response
        formatted_response = self._format_jefe_response(raw_result, jefe_context, task_analysis_result)

        # 4. Log the full interaction (context + final formatted response)
        self.context_manager.add_interaction(jefe_context, formatted_response)

        return formatted_response

    # --- Handler Methods (Single Agent, Parallel Tools Pattern) ---
    async def _handle_simple_assistance(self, context: JefeContext, analysis: TaskAnalysis) -> Dict[str, Any]:
        """Handles tasks deemed 'simple', often by selecting a primary tool."""
        self.logger.info(f"Handling simple assistance for task type: {analysis.task_type.value}")

        # Basic tool selection: use the first suggested tool or a default fallback.
        # Future: Implement more sophisticated primary tool selection based on analysis.
        primary_tool_name = analysis.suggested_tools[0] if analysis.suggested_tools else "code_analyzer_tool" # Example fallback

        tool_to_execute = self.tools.get(primary_tool_name)

        if tool_to_execute:
            self.logger.info(f"Executing primary tool for simple assistance: '{primary_tool_name}'")
            try:
                # Tools are expected to take context and potentially specific kwargs based on analysis
                # For simple assistance, we might pass specific parts of the context or analysis.
                # Example: tool_kwargs = {"error_message": analysis.immediate_issues[0]} if analysis.immediate_issues else {}
                tool_kwargs = {} # Keep it simple for now
                result = await tool_to_execute.execute(context, **tool_kwargs)

                # Adapt tool's raw result to the expected structure for _format_jefe_response
                return {
                    "primary_issue": result.get("data", {}).get("findings", "Issue identified by tool.") if isinstance(result.get("data"), dict) else result.get("data", "Issue identified by tool."),
                    "recommended_action": result.get("data", {}).get("recommendation", "Action suggested by tool.") if isinstance(result.get("data"), dict) else "Action suggested by tool.",
                    "additional_value": result.get("data", {}).get("enhancement", "Further improvements may be possible.") if isinstance(result.get("data"), dict) else "Further improvements may be possible.",
                    "confidence": result.get("confidence", 0.75), # Tool should provide confidence
                    "tools_used": [primary_tool_name],
                    "details": result # Store full tool result for richer formatting or logging
                }
            except Exception as e:
                self.logger.error(f"Error executing tool '{primary_tool_name}' for simple assistance: {e}", exc_info=True)
                return {"primary_issue": f"Error with tool '{primary_tool_name}'.",
                        "recommended_action": "Review tool execution logs and system status.",
                        "additional_value": str(e),
                        "confidence": 0.2,
                        "tools_used": [primary_tool_name]}
        else:
            self.logger.warning(f"Tool '{primary_tool_name}' not found for simple assistance. No action taken.")
            return {"primary_issue": f"Tool '{primary_tool_name}' not available.",
                    "recommended_action": "Verify tool configuration and availability.",
                    "confidence": 0.3,
                    "tools_used": []}

    async def _handle_parallel_analysis(self, context: JefeContext, analysis: TaskAnalysis) -> Dict[str, Any]:
        """Handles 'moderate' complexity tasks by running multiple relevant tools in parallel and synthesizing results."""
        self.logger.info(f"Handling parallel analysis for task type: {analysis.task_type.value}")

        # Select a few relevant tools based on analysis.suggested_tools
        # Future: Implement more sophisticated selection of parallel tools.
        selected_tool_names = analysis.suggested_tools[:3] if analysis.suggested_tools else ["code_analyzer_tool", "documentation_search_tool"] # Example fallback

        tool_execution_tasks = []
        valid_tools_to_execute_names = []
        for tool_name in selected_tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                # Each tool's execute method is a coroutine
                tool_execution_tasks.append(tool.execute(context)) # Assuming tools take JefeContext directly
                valid_tools_to_execute_names.append(tool_name)
            else:
                self.logger.warning(f"Tool '{tool_name}' selected for parallel analysis not found.")

        if not tool_execution_tasks:
            self.logger.warning("No valid tools found or selected for parallel analysis.")
            return {"primary_issue": "No tools available for parallel analysis.",
                    "recommended_action": "Check tool configuration.",
                    "confidence": 0.3, "tools_used": []}

        self.logger.info(f"Executing parallel tools for analysis: {valid_tools_to_execute_names}")
        # `return_exceptions=True` allows us to get results even if some tools fail
        tool_results = await asyncio.gather(*tool_execution_tasks, return_exceptions=True)

        # Synthesize results from parallel tool executions
        synthesized_result = self._synthesize_parallel_results(tool_results, valid_tools_to_execute_names, context, analysis, prefix="Parallel Analysis")
        return synthesized_result

    async def _handle_complex_research(self, context: JefeContext, analysis: TaskAnalysis) -> Dict[str, Any]:
        """Handles 'complex' tasks, potentially involving multi-phase tool execution or deeper research."""
        self.logger.info(f"Handling complex research for task type: {analysis.task_type.value}")

        # Phase 1: Broad exploration with a set of tools
        # Future: Tool selection could be more dynamic based on initial analysis.
        exploration_tool_names = analysis.suggested_tools if analysis.suggested_tools else ["web_search_tool", "documentation_search_tool", "code_analyzer_tool"]
        if not exploration_tool_names: # Ensure there's at least some default
             exploration_tool_names = ["web_search_tool"]


        exploration_tasks = []
        valid_exploration_tools = []
        for tool_name in exploration_tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                exploration_tasks.append(tool.execute(context)) # Pass current context
                valid_exploration_tools.append(tool_name)
            else:
                self.logger.warning(f"Exploration tool '{tool_name}' for complex research not found.")

        if not exploration_tasks:
            return {"primary_issue": "No tools available for initial exploration in complex research.",
                    "recommended_action": "Check tool configuration.", "confidence": 0.3, "tools_used": []}

        self.logger.info(f"Executing complex research (exploration phase) with tools: {valid_exploration_tools}")
        exploration_results = await asyncio.gather(*exploration_tasks, return_exceptions=True)

        # Synthesize exploration results
        # This synthesis could identify key findings or areas for a deeper dive.
        exploration_synthesis = self._synthesize_parallel_results(exploration_results, valid_exploration_tools, context, analysis, prefix="Exploration Summary")

        # Placeholder for Phase 2: Deeper dive or focused tool execution based on Phase 1.
        # For now, complex research returns the synthesis of the exploration phase.
        # Future: Could involve using an LLM to analyze exploration_synthesis and plan next steps,
        # or select/configure more specialized tools.

        return {"primary_issue": exploration_synthesis.get('primary_issue', "Complex research exploration complete."),
                "recommended_action": exploration_synthesis.get('recommended_action', "Review exploration findings for further action."),
                "additional_value": exploration_synthesis.get('additional_value', "Detailed results from exploration tools are available."),
                "confidence": exploration_synthesis.get('confidence', 0.65),
                "tools_used": valid_exploration_tools, # Tools used in the exploration phase
                "details": {"exploration_results_summary": exploration_synthesis} # Attach the synthesis
               }

    # --- Helper Methods ---
    def _format_jefe_response(self, raw_tool_result: Dict[str, Any], context: JefeContext, task_analysis: TaskAnalysis) -> Dict[str, Any]:
        """
        Formats the raw result from handler methods into a standardized response structure.
        This method aims to create a user-friendly and informative response.
        """
        # Extract key pieces of information from the raw tool result
        issue = raw_tool_result.get('primary_issue', 'Analysis complete. No specific issue highlighted.')
        solution = raw_tool_result.get('recommended_action', 'Review the provided information and proceed as appropriate.')
        enhancement = raw_tool_result.get('additional_value', '') # Optional additional info

        # Construct a user-friendly text response.
        # This can be made more sophisticated, perhaps using an LLM for summarization/formatting if needed.
        formatted_response_parts = []
        if issue: formatted_response_parts.append(f"🔧 Finding: {issue}")
        if solution: formatted_response_parts.append(f"💡 Suggestion: {solution}")
        if enhancement: formatted_response_parts.append(f"⚡ Additionally: {enhancement}")

        if not formatted_response_parts: # Fallback if no specific parts were generated
            formatted_response_text = "Jefe has processed your context. No specific actions suggested at this time."
            if raw_tool_result.get("tools_used"):
                formatted_response_text += f" (Tools utilized: {', '.join(raw_tool_result['tools_used'])})."
        else:
            formatted_response_text = "\n".join(formatted_response_parts)

        # Standard response structure
        return {
            'status': "success", # Assuming success if we got this far, errors handled in tool execution
            'content': formatted_response_text, # User-facing formatted response
            'confidence': raw_tool_result.get('confidence', 0.8), # Overall confidence from the handling method
            'task_type_analysis': str(task_analysis.task_type.value), # Analyzed task type
            'complexity_analysis': task_analysis.complexity, # Analyzed complexity
            'priority_analysis': task_analysis.priority, # Analyzed priority
            'tools_used': raw_tool_result.get('tools_used', []), # List of tools that contributed
            'raw_tool_outputs': raw_tool_result.get('details') # Optional: pass raw tool outputs for debugging or UI
        }

    def _synthesize_parallel_results(self, results: list, tool_names: List[str], context: JefeContext, analysis: TaskAnalysis, prefix: str = "") -> Dict[str, Any]:
        """
        Basic synthesis of results from multiple tools run in parallel.
        This is a placeholder and needs to be made much more sophisticated.
        Ideally, an LLM would synthesize these results based on the initial query and context.
        """
        self.logger.info(f"Synthesizing {len(results)} parallel results for {analysis.task_type.value}. Prefix: '{prefix}'")

        successful_outputs: List[str] = []
        error_outputs: List[str] = []
        all_tool_details = [] # Store individual tool results for potential inclusion
        combined_confidence = 0.0
        num_successful_tools = 0

        for i, res in enumerate(results):
            tool_name = tool_names[i] if i < len(tool_names) else "unknown_tool"

            if isinstance(res, Exception):
                self.logger.error(f"Error from tool '{tool_name}' during parallel execution: {res}", exc_info=res)
                error_outputs.append(f"Error - {tool_name}: {str(res)[:150]}") # Truncate error message
                all_tool_details.append({"tool": tool_name, "status": "error", "output": str(res)})
                continue

            if not isinstance(res, dict):
                self.logger.warning(f"Unexpected result type from tool '{tool_name}': {type(res)}. Expected dict.")
                error_outputs.append(f"Non-dict result from {tool_name}.")
                all_tool_details.append({"tool": tool_name, "status": "unexpected_type", "output": str(res)[:200]})
                continue

            all_tool_details.append({"tool": tool_name, "status": res.get("status", "unknown"), "output": res}) # Store full result

            if res.get("status") == "success":
                num_successful_tools += 1
                # Prefer 'content' if available, else 'data', else stringify 'data' if it's a dict.
                content = res.get("content")
                if content is None:
                    data_payload = res.get("data")
                    if isinstance(data_payload, dict):
                        # Try to get a meaningful summary from data if it's a dict
                        content = data_payload.get("summary", data_payload.get("findings", str(data_payload)[:150]))
                    else:
                        content = str(data_payload)[:150] if data_payload is not None else ""

                successful_outputs.append(f"Tool '{tool_name}': {content}")
                combined_confidence += res.get("confidence", 0.5) # Use 0.5 as neutral if no confidence
            else:
                error_outputs.append(f"Failed - {tool_name}: {res.get('error', res.get('message', 'Unknown error'))[:150]}")

        # Naive synthesis logic
        final_issue_parts = []
        final_solution_parts = []
        final_enhancement_parts = []

        if successful_outputs:
            final_issue_parts.append(f"{prefix} Key Insights Gathered:")
            for i, output_str in enumerate(successful_outputs):
                 final_issue_parts.append(f"  - {output_str}")
        if error_outputs:
            final_issue_parts.append(f"{prefix} Issues Encountered During Analysis:")
            for i, err_str in enumerate(error_outputs):
                final_issue_parts.append(f"  - {err_str}")

        if not final_issue_parts:
            final_issue_parts.append(f"{prefix} Analysis performed, but no specific textual output to synthesize.")

        # For now, recommended_action and additional_value are generic
        final_solution_parts.append("Review the detailed findings from each tool (if available in 'details').")
        if num_successful_tools < len(tool_names):
             final_solution_parts.append("Some tools encountered errors; check logs for details.")


        avg_confidence = (combined_confidence / num_successful_tools) if num_successful_tools > 0 else 0.3

        return {
            "primary_issue": "\n".join(final_issue_parts),
            "recommended_action": "\n".join(final_solution_parts),
            "additional_value": "\n".join(final_enhancement_parts) if final_enhancement_parts else "Further investigation may be needed based on detailed outputs.",
            "confidence": round(avg_confidence, 2),
            "tools_used": tool_names, # All tools attempted
            "details": all_tool_details # Store individual raw results
        }

    def _load_tools(self) -> Dict[str, 'BaseTool']:
        """
        Dynamically loads tools from the 'multi-agent-system/src/tools' directory.
        Tool files should be named like 'some_tool_name_tool.py' and contain a class
        named 'SomeToolNameTool' that inherits from BaseTool.
        The key in the returned dictionary will be 'some_tool_name'.
        """
        # Determine the path to the 'tools' directory relative to this file's location (agents directory)
        # current_dir (agents) -> parent (src) -> tools_dir (tools)
        current_file_path = os.path.abspath(__file__)
        agents_dir = os.path.dirname(current_file_path)
        src_dir = os.path.dirname(agents_dir)
        tools_dir = os.path.join(src_dir, "tools")

        loaded_tools: Dict[str, 'BaseTool'] = {}

        if not os.path.isdir(tools_dir):
            self.logger.warning(f"Tools directory not found at expected path: {tools_dir}. No tools will be loaded.")
            return loaded_tools

        for filename in os.listdir(tools_dir):
            if filename.endswith("_tool.py") and not filename.startswith("__"):
                module_name_fs = filename[:-3] # Filename without .py (e.g., "code_analysis_tool")

                # Convert snake_case filename to CamelCase class name
                # e.g., "code_analysis_tool" -> "CodeAnalysisTool"
                class_name_parts = [part.capitalize() for part in module_name_fs.split('_')]
                class_name = "".join(class_name_parts)

                # Determine the tool key for the dictionary (e.g., "code_analysis")
                tool_key_parts = module_name_fs.split('_')
                if tool_key_parts[-1] == "tool": # Remove "_tool" suffix
                    tool_key = "_".join(tool_key_parts[:-1])
                else:
                    tool_key = module_name_fs # Should not happen if naming convention is followed

                try:
                    # Construct the full module path for importlib relative to 'src'
                    # e.g., "multi-agent-system.src.tools.code_analysis_tool"
                    # The package argument to import_module should be the parent package
                    # from which the relative import is performed.
                    # If JefeAgent is in src.agents.jefe_agent, then "..tools" refers to src.tools
                    module = importlib.import_module(f"..tools.{module_name_fs}", package=__name__)
                    ToolClass = getattr(module, class_name, None)

                    if ToolClass and issubclass(ToolClass, BaseTool): # type: ignore # BaseTool might be string due to TYPE_CHECKING
                        # Instantiate the tool. Assumes tool constructor takes api_manager.
                        # Tools should ideally set their own name and description upon instantiation
                        # or have static methods to retrieve them before instantiation if needed by BaseTool.
                        tool_instance = ToolClass(api_manager=self.api_manager)

                        # Use tool_instance.tool_name if available and consistently set by tools,
                        # otherwise, fallback to tool_key derived from filename.
                        # For now, using tool_key is more robust for dynamic loading.
                        loaded_tools[tool_key] = tool_instance
                        self.logger.info(f"Successfully loaded tool: '{tool_key}' (Class: {class_name}) from {filename}")
                    else:
                        self.logger.warning(f"Could not load class '{class_name}' from '{filename}' or it's not a subclass of BaseTool.")
                except ImportError as e:
                    self.logger.error(f"Error importing tool module '{module_name_fs}' from '{filename}': {e}", exc_info=True)
                except Exception as e: # Catch other potential errors during instantiation
                    self.logger.error(f"Error instantiating tool '{class_name}' from '{module_name_fs}': {e}", exc_info=True)

        return loaded_tools

if __name__ == '__main__':
    # This block is for basic conceptual demonstration and testing.
    # It requires APIManager, JefeContext, BaseTool, and potentially some mock tools to be available.

    # Define MockAPIManager and MockJefeContext if they are not importable or for isolated testing
    # This ensures `python path/to/jefe_agent.py` can run with minimal setup for basic checks.

    # Fallback MockAPIManager if the real one isn't easily available for this direct script run
    if 'APIManager' not in globals() or APIManager.__module__ != 'src.utils.api_manager':
        class MockAPIManager: # type: ignore
            def __init__(self, *args, **kwargs):
                self.logger = get_logger("MockAPIManager_JefeAgentTest")
                self.logger.info("Instantiated MockAPIManager for JefeAgent test.")
            async def call_llm_service(self, *args, **kwargs) -> Dict[str, Any]:
                self.logger.info(f"MockAPIManager.call_llm_service called with: {args}, {kwargs}")
                return {"status": "success", "content": "Mocked LLM response."}
    else: # If the real APIManager was imported, use it (or a more specific mock if needed)
        MockAPIManager = APIManager # type: ignore

    # Fallback MockBaseTool (very basic)
    if TYPE_CHECKING: # This helps type checkers but won't run at runtime if BaseTool is not imported
        from ..tools.base_tool import BaseTool as ActualBaseTool
        MockBaseTool = ActualBaseTool
    else: # Runtime fallback
        class MockBaseTool:
            def __init__(self, tool_name, tool_description, api_manager=None):
                self.tool_name = tool_name
                self.tool_description = tool_description
                self.api_manager = api_manager
                self.logger = get_logger(f"MockTool.{tool_name}")
            async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
                self.logger.info(f"MockTool {self.tool_name} execute called with context: {context.summarize(50,50)}, kwargs: {kwargs}")
                return {"status": "success", "data": f"Mock result from {self.tool_name}", "confidence": 0.9}
            def get_description(self) -> Dict[str, str]: return {"name": self.tool_name, "description": self.tool_description}
        # Make BaseTool refer to MockBaseTool if the real one wasn't imported due to TYPE_CHECKING
        if 'BaseTool' not in globals(): BaseTool = MockBaseTool # type: ignore


    async def main_jefe_agent_test():
        print("--- JefeAgent Conceptual Test ---")

        # Instantiate with a mock APIManager
        mock_api_manager = MockAPIManager() # type: ignore

        # To test _load_tools, we might need to create dummy tool files in a temporary tools dir
        # or mock os.listdir, importlib.import_module, etc.
        # For simplicity here, we'll assume _load_tools might return an empty dict if no tools dir/files.
        # Or, we can manually populate self.tools with mock tools for this test.

        agent_config = {"default_model": "jefe-model-v1"}
        jefe = JefeAgent(agent_name="TestJefe001", api_manager=mock_api_manager, config=agent_config)

        # Manually add a mock tool for testing internal logic if dynamic loading is complex to set up here
        class MockCodeAnalyzerTool(MockBaseTool): # type: ignore
            def __init__(self, api_manager_instance): # Renamed to avoid conflict
                super().__init__("code_analyzer_tool", "Analyzes code for issues.", api_manager_instance)
            async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
                findings = "Found potential syntax error in screen content." if "error" in context.screen_content.lower() else "Code seems okay at first glance."
                return {"status": "success", "data": {"findings": findings, "recommendation": "Review highlighted lines."}, "confidence": 0.8}

        if not jefe.tools.get("code_analyzer"): # If not loaded dynamically
             jefe.tools["code_analyzer"] = MockCodeAnalyzerTool(mock_api_manager) # type: ignore
             jefe.logger.info("Manually added MockCodeAnalyzerTool for testing.")


        print(f"\nJefeAgent Capabilities: {jefe.get_capabilities()}")

        # Test 1: Simple text query (adapted to JefeContext)
        print("\n--- Test 1: Simple Text Query ---")
        query_data_1 = {"prompt": "Explain Python list comprehensions."}
        response_1 = await jefe.process_query(query_data_1)
        print("Response 1:")
        for key, value in response_1.items():
            print(f"  {key}: {str(value)[:200] + '...' if len(str(value)) > 200 else value}")


        # Test 2: Real-time input scenario - simple complexity
        print("\n--- Test 2: Real-time Input (Simple Complexity) ---")
        context_2 = JefeContext(
            screen_content="def hello():\n  print('world'", # Missing parenthesis
            audio_transcript="User: I think I have a syntax error here, what's wrong?",
            current_ide="VSCode",
            programming_language="Python",
            error_messages=["SyntaxError: unexpected EOF while parsing"]
        )
        response_2 = await jefe.process_realtime_input(context_2)
        print("Response 2:")
        for key, value in response_2.items():
            print(f"  {key}: {str(value)[:200] + '...' if len(str(value)) > 200 else value}")

        # Test 3: Real-time input - moderate (triggering parallel if tools were diverse)
        print("\n--- Test 3: Real-time Input (Moderate Complexity - Parallel Tools Simulation) ---")
        # Add another mock tool for parallel test
        class MockDocSearchTool(MockBaseTool): # type: ignore
             def __init__(self, api_manager_instance):
                super().__init__("documentation_search_tool", "Searches documentation.", api_manager_instance)
             async def execute(self, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
                return {"status": "success", "data": {"summary": f"Found docs related to: {context.programming_language or 'general topics'}"}, "confidence": 0.7}

        if not jefe.tools.get("documentation_search"):
            jefe.tools["documentation_search"] = MockDocSearchTool(mock_api_manager) # type: ignore
            jefe.logger.info("Manually added MockDocSearchTool for testing.")
            # Update capabilities if we added a tool manually after init
            # This is a bit of a hack for testing; normally tools are loaded at init.
            jefe.task_analyzer.task_type_keywords[JefeTaskType.KNOWLEDGE_QUERY] = ["explain"] # Add keyword for this tool for test
            # Make sure suggested_tools in TaskAnalysis can pick this up for some query

        context_3 = JefeContext(
            screen_content="import pandas as pd\ndf = pd.DataFrame() # User is working with pandas",
            audio_transcript="User: Explain how pandas DataFrames work and how to optimize their memory usage.",
            programming_language="Python",
            project_type="Data Analysis"
        )
        response_3 = await jefe.process_realtime_input(context_3)
        print("Response 3 (Parallel Analysis Simulation):")
        for key, value in response_3.items():
            print(f"  {key}: {str(value)[:200] + '...' if len(str(value)) > 200 else value}")

        # Test 4: Context Compression (if manager has interactions)
        print("\n--- Test 4: Context Compression via JefeContextManager ---")
        if jefe.context_manager.get_full_conversation_history():
            # Ensure there are enough interactions to attempt compression based on manager's thresholds
            jefe.context_manager.max_history_len_before_compression = 1
            jefe.context_manager.recent_interactions_to_keep = 0
            await jefe.context_manager._compress_old_context() # Call directly for test
            print(f"Compressed Memory: {jefe.context_manager.get_compressed_memory_summary()}")
        else:
            print("Skipping compression test as no interactions were logged to context_manager.")


    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_jefe_agent_test())
    print("\n--- JefeAgent conceptual test finished ---")

# TODO:
# - Implement sophisticated tool selection logic in _select_primary_tool_name, _select_parallel_tool_names, etc.
# - Implement robust synthesis logic in _synthesize_parallel_results and for complex research.
# - Flesh out _update_context_with_findings for complex research.
# - Refine interaction between JefeTaskAnalyzer's suggestions and tool selection here.
# - Ensure BaseTool and concrete tool __init__ signatures are compatible with _load_tools.
#   (e.g., tools define their own name/desc or _load_tools infers it more robustly).
# - Add actual tool implementations in the tools directory.
# - More comprehensive error handling and state management.
