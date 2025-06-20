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

        # raw_result: Dict[str, Any] # This will be changed by the handlers
        # 2. Determine handling strategy based on complexity
        # The handler methods will now return a List[Dict[str, Any]] of raw tool results
        raw_tool_results_list: List[Dict[str, Any]]
        if task_analysis_result.complexity == "simple":
            raw_tool_results_list = await self._handle_simple_assistance(jefe_context, task_analysis_result)
        elif task_analysis_result.complexity == "moderate":
            raw_tool_results_list = await self._handle_parallel_analysis(jefe_context, task_analysis_result)
        else: # "complex"
            raw_tool_results_list = await self._handle_complex_research(jefe_context, task_analysis_result)

        # 3. Format the list of raw tool results into a user-facing response
        # The _format_jefe_response will be updated in a subsequent step to accept List[Dict[str, Any]]
        formatted_response = self._format_jefe_response(raw_tool_results_list, jefe_context, task_analysis_result) # type: ignore

        # 4. Log the full interaction (context + final formatted response)
        self.context_manager.add_interaction(jefe_context, formatted_response)

        return formatted_response

    # --- Handler Methods (Single Agent, Parallel Tools Pattern) ---
    async def _handle_simple_assistance(self, context: JefeContext, analysis: TaskAnalysis) -> List[Dict[str, Any]]:
        """
        Handles tasks deemed 'simple', often by selecting a primary tool.
        Returns a list containing a single raw tool result dictionary or an error dictionary.
        """
        self.logger.info(f"Handling simple assistance for task type: {analysis.task_type.value}")

        # Use tool keys (e.g., "code_analyzer") not "code_analyzer_tool"
        primary_tool_name = analysis.suggested_tools[0] if analysis.suggested_tools else "code_analyzer"

        tool_to_execute = self.tools.get(primary_tool_name)

        if tool_to_execute:
            self.logger.info(f"Executing primary tool for simple assistance: '{primary_tool_name}'")
            try:
                tool_kwargs = {}
                result = await tool_to_execute.execute(context, **tool_kwargs)
                # Ensure tool_used is in the result, tools should ideally do this themselves.
                if 'tool_used' not in result: result['tool_used'] = primary_tool_name
                return [result]
            except Exception as e:
                self.logger.error(f"Error executing tool '{primary_tool_name}' for simple assistance: {e}", exc_info=True)
                return [{"status": "error", "tool_used": primary_tool_name, "error": f"Exception during tool execution: {str(e)}"}]
        else:
            self.logger.warning(f"Tool '{primary_tool_name}' not found for simple assistance. No action taken.")
            return [{"status": "error", "tool_used": primary_tool_name, "error": f"Tool '{primary_tool_name}' not available."}]

    async def _handle_parallel_analysis(self, context: JefeContext, analysis: TaskAnalysis) -> List[Dict[str, Any]]:
        """
        Handles 'moderate' complexity tasks by running multiple relevant tools in parallel.
        Returns a list of raw tool result dictionaries or error dictionaries.
        """
        self.logger.info(f"Handling parallel analysis for task type: {analysis.task_type.value}")

        selected_tool_names = analysis.suggested_tools[:3] if analysis.suggested_tools else ["code_analyzer", "documentation_search"] # Use tool keys

        tool_execution_tasks = []
        valid_tools_to_execute_names = []
        for tool_name in selected_tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                tool_execution_tasks.append(tool.execute(context))
                valid_tools_to_execute_names.append(tool_name)
            else:
                self.logger.warning(f"Tool '{tool_name}' selected for parallel analysis not found.")
                # Add an error placeholder if a suggested tool is missing
                valid_tools_to_execute_names.append(tool_name) # Keep for indexing against results
                tool_execution_tasks.append(asyncio.create_task(self._mock_missing_tool_error(tool_name)))


        if not tool_execution_tasks: # Should only happen if selected_tool_names was empty initially
            self.logger.warning("No tools were selected or available for parallel analysis.")
            return [{"status": "error", "tool_used": "parallel_analysis_handler", "error": "No tools selected for parallel analysis."}]

        self.logger.info(f"Executing parallel tools for analysis: {valid_tools_to_execute_names}")
        tool_results_raw = await asyncio.gather(*tool_execution_tasks, return_exceptions=True)

        processed_results: List[Dict[str, Any]] = []
        for i, res_or_exc in enumerate(tool_results_raw):
            tool_name = valid_tools_to_execute_names[i]
            if isinstance(res_or_exc, Exception):
                self.logger.error(f"Tool '{tool_name}' in parallel analysis raised an exception: {res_or_exc}", exc_info=res_or_exc)
                processed_results.append({"status": "error", "tool_used": tool_name, "error": str(res_or_exc)})
            elif isinstance(res_or_exc, dict):
                if 'tool_used' not in res_or_exc: # Ensure tool_used is present
                    res_or_exc['tool_used'] = tool_name
                processed_results.append(res_or_exc)
            else:
                 self.logger.warning(f"Tool '{tool_name}' in parallel analysis returned unexpected type: {type(res_or_exc)}")
                 processed_results.append({"status": "error", "tool_used": tool_name, "error": "Unexpected result type from tool."})
        return processed_results

    async def _handle_complex_research(self, context: JefeContext, analysis: TaskAnalysis) -> List[Dict[str, Any]]:
        """
        Handles 'complex' tasks, potentially involving multi-phase tool execution.
        Returns a list of raw tool result dictionaries or error dictionaries.
        """
        self.logger.info(f"Handling complex research for task type: {analysis.task_type.value}")

        exploration_tool_names = analysis.suggested_tools if analysis.suggested_tools else ["web_search", "documentation_search"] # Use tool keys
        if not exploration_tool_names: exploration_tool_names = ["web_search"] # Default

        exploration_tasks = []
        valid_exploration_tools = []
        for tool_name in exploration_tool_names:
            tool = self.tools.get(tool_name)
            if tool:
                exploration_tasks.append(tool.execute(context))
                valid_exploration_tools.append(tool_name)
            else:
                self.logger.warning(f"Exploration tool '{tool_name}' for complex research not found.")
                valid_exploration_tools.append(tool_name) # Keep for indexing
                exploration_tasks.append(asyncio.create_task(self._mock_missing_tool_error(tool_name)))

        if not exploration_tasks: # Should only happen if exploration_tool_names was empty
            return [{"status": "error", "tool_used": "complex_research_handler", "error": "No tools selected for complex research."}]

        self.logger.info(f"Executing complex research (exploration phase) with tools: {valid_exploration_tools}")
        exploration_results_raw = await asyncio.gather(*exploration_tasks, return_exceptions=True)

        processed_exploration_results: List[Dict[str, Any]] = []
        for i, res_or_exc in enumerate(exploration_results_raw):
            tool_name = valid_exploration_tools[i]
            if isinstance(res_or_exc, Exception):
                self.logger.error(f"Tool '{tool_name}' in complex research raised an exception: {res_or_exc}", exc_info=res_or_exc)
                processed_exploration_results.append({"status": "error", "tool_used": tool_name, "error": str(res_or_exc)})
            elif isinstance(res_or_exc, dict):
                if 'tool_used' not in res_or_exc: res_or_exc['tool_used'] = tool_name
                processed_exploration_results.append(res_or_exc)
            else:
                 self.logger.warning(f"Tool '{tool_name}' in complex research returned unexpected type: {type(res_or_exc)}")
                 processed_exploration_results.append({"status": "error", "tool_used": tool_name, "error": "Unexpected result type from tool."})

        # For now, complex research returns the list of raw results from the exploration phase.
        # Future: Could involve further phases and more sophisticated synthesis logic.
        return processed_exploration_results

    async def _mock_missing_tool_error(self, tool_name: str) -> Dict[str, Any]:
        """Helper to create a standard error dict for a missing tool for asyncio.gather."""
        return {"status": "error", "tool_used": tool_name, "error": f"Tool '{tool_name}' not found/available."}

    # --- Helper Methods ---
    def _format_jefe_response(self, raw_tool_results: List[Dict[str, Any]], context: JefeContext, task_analysis: TaskAnalysis) -> Dict[str, Any]:
        """
        Formats the synthesized result from tools into a standardized user-facing response structure.
        """
        # 1. Synthesize the raw tool results using the main synthesis method
        synthesized_info = self._synthesize_tool_results(raw_tool_results, context, task_analysis.task_type)

        # 2. Extract key pieces for the user-facing text message
        issue = synthesized_info.get('primary_issue', 'Analysis complete. No specific issue highlighted.')
        solution = synthesized_info.get('recommended_action', 'Review the provided information and proceed as appropriate.')
        enhancement = synthesized_info.get('additional_value', '') # Optional

        # 3. Construct the "🔧💡⚡" formatted text string
        formatted_response_parts = []
        if issue: formatted_response_parts.append(f"🔧 Finding: {issue}")
        if solution: formatted_response_parts.append(f"💡 Suggestion: {solution}")
        if enhancement: formatted_response_parts.append(f"⚡ Additionally: {enhancement}")

        if not formatted_response_parts:
            if synthesized_info.get('confidence', 0) < 0.2: # Likely from _handle_all_tools_failed
                 formatted_response_text = "Jefe encountered issues processing your request. " + issue # 'issue' will contain error summary
            else:
                formatted_response_text = "Jefe has processed your context. No specific actions suggested at this time."
                if synthesized_info.get("tools_used"):
                    formatted_response_text += f" (Tools utilized: {', '.join(synthesized_info['tools_used'])})."
        else:
            formatted_response_text = "\n".join(formatted_response_parts)

        # 4. Construct the final response dictionary
        final_response = {
            'status': "success" if synthesized_info.get('confidence', 0) >= 0.2 else "error", # success unless all tools failed
            'content': formatted_response_text,
            'confidence': synthesized_info.get('confidence', 0.5),
            'task_type': str(task_analysis.task_type.value), # From original analysis
            'complexity': task_analysis.complexity,     # From original analysis
            'priority': task_analysis.priority,         # From original analysis
            'identified_issues_summary': synthesized_info.get('primary_issue'), # From synthesis
            'recommended_actions_summary': synthesized_info.get('recommended_action'), # From synthesis
            'enhancement_tips_summary': synthesized_info.get('additional_value'), # From synthesis
            'tools_used': synthesized_info.get('tools_used', []),
            'raw_tool_results': synthesized_info.get('details', {}).get('raw_tool_outputs', raw_tool_results) # Ensure 'details' exists
            # 'context_updates': synthesized_info.get('context_updates', {}) # If we add context updates
        }
        return final_response

    # --- New Synthesis Helper Methods ---
    def _identify_primary_issue(self, results: List[Dict[str, Any]], context: JefeContext) -> str:
        """Identifies the most pressing issue from successful tool results or context."""
        # This method now expects `results` to be a list of SUCCESSFUL tool results.
        for res in results:
            # Look for specific keys that tools might use to highlight primary findings
            primary_finding = res.get("primary_finding") or res.get("findings") or \
                              (res.get("data", {}).get("findings") if isinstance(res.get("data"), dict) else None)
            if primary_finding and isinstance(primary_finding, str) and len(primary_finding) > 10:
                return f"Key finding from {res.get('tool_used', 'a tool')}: {(primary_finding.split('.')[0] or primary_finding.splitlines()[0])[:200]}"

        if context.error_messages: # Check original context if no tool highlighted an issue
            return f"User is facing an error: {context.error_messages[0][:150]}"

        return "General context analysis performed; no single primary issue pinpointed by tools."

    def _determine_best_action(self, results: List[Dict[str, Any]], context: JefeContext) -> str:
        """Determines the most relevant next action or recommendation from successful tool results."""
        # This method now expects `results` to be a list of SUCCESSFUL tool results.
        for res in results:
            recommendation = res.get("recommendation") or \
                             (res.get("data", {}).get("recommendation") if isinstance(res.get("data"), dict) else None)
            if recommendation and isinstance(recommendation, str) and len(recommendation) > 10:
                return recommendation[:300]

        return "Review the detailed outputs from the executed tools for specific suggestions."

    def _extract_enhancement_tip(self, results: List[Dict[str, Any]], context: JefeContext) -> str:
        """Extracts a relevant enhancement tip or additional advice from successful tool results."""
        # This method now expects `results` to be a list of SUCCESSFUL tool results.
        for res in results:
            enhancement = res.get("additional_value") or \
                          (res.get("data", {}).get("enhancement") if isinstance(res.get("data"), dict) else None) or \
                          res.get("tip")
            if enhancement and isinstance(enhancement, str) and len(enhancement) > 5:
                return enhancement[:200]

        if context.programming_language:
            return f"Consider exploring advanced features or libraries for {context.programming_language} relevant to your task."
        return "Reviewing documentation for used technologies can often reveal further optimization opportunities."

    def _calculate_synthesis_confidence(self, all_tool_results: List[Dict[str, Any]]) -> float:
        """
        Calculates an overall confidence score based on ALL tool results (successes and failures).
        The `successful_results` list is used by other helpers, but this one needs all to assess overall reliability.
        """
        if not all_tool_results: return 0.1 # No tools ran or no results

        successful_confidences = [
            res.get("confidence", 0.5) for res in all_tool_results
            if isinstance(res, dict) and res.get("status") == "success" and isinstance(res.get("confidence"), (float, int))
        ]

        num_total_tools = len(all_tool_results)
        num_successful_tools = len(successful_confidences)

        if num_successful_tools == 0: # All tools failed or no successful tools reported confidence
            return 0.1

        # Average confidence of successful tools
        avg_successful_confidence = sum(successful_confidences) / num_successful_tools

        # Modulate by the ratio of successful tools
        success_ratio = num_successful_tools / num_total_tools

        # Weighted confidence: higher if more tools succeeded and their individual confidences were high.
        # Example weighting: 70% from avg successful confidence, 30% from success ratio.
        # Adjust weights as needed.
        final_confidence = (avg_successful_confidence * 0.7) + (success_ratio * 0.3)

        return round(final_confidence, 2)

    def _handle_all_tools_failed(self, error_results: List[Dict[str, Any]], context: JefeContext) -> Dict[str, Any]:
        """Formats a response when all tools fail or return errors."""
        self.logger.warning(f"All tools failed or returned errors. Number of error results: {len(error_results)}")
        error_summary_parts = []
        tools_that_failed = set()
        for res in error_results[:3]: # Limit to first 3 errors for brevity
            tool_name = res.get('tool_used', 'UnknownTool')
            tools_that_failed.add(tool_name)
            err_msg = res.get('error', 'Undescribed failure')
            error_summary_parts.append(f"{tool_name}: {str(err_msg)[:100]}")

        error_summary_str = "; ".join(error_summary_parts)
        if not error_summary_str and error_results:
            error_summary_str = "Multiple tools encountered issues."
        elif not error_results:
            error_summary_str = "No tool results to process, though failure was indicated."

        return {
            "primary_issue": f"Unable to complete analysis due to tool failures. Errors: {error_summary_str}",
            "recommended_action": "The system encountered issues with its internal tools. Please try rephrasing your request, or check tool configurations and external service status.",
            "additional_value": "If the problem persists, reviewing system logs for detailed error messages from tools might be helpful.",
            "confidence": 0.1,
            "tools_used": list(tools_that_failed),
            "details": {"raw_tool_outputs": error_results}
        }

    def _extract_context_updates(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Placeholder for future logic to extract context updates from successful tool results."""
        # This method now expects `results` to be a list of SUCCESSFUL tool results.
        return {}

    def _synthesize_tool_results(self, tool_results_raw: List[Dict[str, Any]], context: JefeContext, task_type: JefeTaskType) -> Dict[str, Any]:
        """
        Synthesizes raw results from one or more tools into a coherent summary dictionary.
        """
        self.logger.info(f"Synthesizing {len(tool_results_raw)} raw tool results for task type: {task_type.value if hasattr(task_type, 'value') else task_type}")

        # `tool_results_raw` is already a list of dicts (results or errors) from handlers.
        processed_tool_results = tool_results_raw

        executed_tool_names = [res.get("tool_used", "unknown_tool") for res in processed_tool_results if isinstance(res, dict)]

        successful_results = [res for res in processed_tool_results if isinstance(res, dict) and res.get("status") == "success"]

        # If no tools ran at all (e.g. handler returned empty list for some reason)
        if not processed_tool_results:
             self.logger.warning("Synthesizing results, but no tool results were provided to _synthesize_tool_results.")
             return {
                "primary_issue": "No tool activity was initiated or reported.",
                "recommended_action": "This may indicate an issue before tool selection or execution.",
                "additional_value": "", "confidence": 0.05, "tools_used": [], "details": {"raw_tool_outputs": []}
            }

        # If all tools that ran failed or returned errors
        if not successful_results:
            return self._handle_all_tools_failed(processed_tool_results, context)

        # If we have at least one successful result, proceed with synthesis
        primary_issue = self._identify_primary_issue(successful_results, context)
        best_action = self._determine_best_action(successful_results, context)
        enhancement_tip = self._extract_enhancement_tip(successful_results, context)
        # Pass ALL processed results (successes and failures) to calculate overall confidence
        confidence = self._calculate_synthesis_confidence(processed_tool_results)

        # context_updates = self._extract_context_updates(successful_results) # For future use

        final_tools_used = list(set(name for name in executed_tool_names if name and name != "unknown_tool"))
        if not final_tools_used and any(name == "unknown_tool" for name in executed_tool_names):
             final_tools_used = ["unknown_tool (attempted)"]


        return {
            "primary_issue": primary_issue,
            "recommended_action": best_action,
            "additional_value": enhancement_tip,
            "confidence": confidence,
            "tools_used": final_tools_used,
            # "context_updates": context_updates, # For future use
            "details": {"raw_tool_outputs": processed_tool_results} # Store all processed results
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

    # Using the actual APIManager for the test, but tools' execute will be mocked.
    # This means APIManager won't actually be called by mocked tools.
    # If APIManager is needed for context compression tests, it can be the real one.

    # Minimal MockAPIManager if the real one causes issues or for extreme isolation
    # For this test, we primarily rely on mocking tool.execute, so APIManager's role in tool calls is bypassed.
    class TestMockAPIManager(APIManager):
        def __init__(self, *args, **kwargs):
            super().__init__(service_configs={"test_service": {"api_key": "dummy", "base_url": "dummy"}}, *args, **kwargs) # Ensure base init is okay
            self.logger = get_logger("TestMockAPIManager_JefeAgent")
            self.logger.info("Instantiated TestMockAPIManager for JefeAgent test.")

        async def call_llm_service(self, *args, **kwargs) -> Dict[str, Any]:
            # This might be called by context_manager for compression, so provide a basic mock response.
            self.logger.info(f"TestMockAPIManager.call_llm_service called with: {args}, {kwargs}. This is likely for context compression.")
            return {"status": "success", "content": "Mocked LLM response for compression."}

    async def main_jefe_agent_test():
        print("--- JefeAgent Integration Test with Mocked Tools & Enhanced Synthesis ---")

        # Instantiate with a mock APIManager (or real if it's simple enough for setup)
        # The key is that tools' .execute() will be mocked, so APIManager calls from tools are bypassed.
        test_api_manager = TestMockAPIManager()

        agent_config = {"default_model": "jefe-model-v1"}
        jefe_agent = JefeAgent(agent_name="TestJefe001", api_manager=test_api_manager, config=agent_config)

        # --- Tool Mocking Setup ---
        # Store original execute methods to restore later if needed (optional for script end)
        original_tools_execute = {}
        tool_keys_to_mock = ["code_analyzer", "debug_assistant", "architecture_advisor", "performance_optimizer"]

        for tool_key in tool_keys_to_mock:
            if tool_key in jefe_agent.tools:
                original_tools_execute[tool_key] = jefe_agent.tools[tool_key].execute
            else:
                jefe_agent.logger.warning(f"Tool '{tool_key}' not found in JefeAgent's loaded tools for mocking. Test might be incomplete.")

        # Define mock execute functions
        async def mock_code_analyzer_execute(context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "tool_used": "code_analyzer",
                    "primary_finding": "Minor syntax error in loop.",
                    "recommendation": "Fix syntax at line 10.", "confidence": 0.9}

        async def mock_debug_assistant_execute(context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "tool_used": "debug_assistant",
                    "primary_finding": "Null pointer exception risk.",
                    "recommendation": "Add null check before accessing object.", "confidence": 0.8}

        async def mock_arch_advisor_execute(context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "tool_used": "architecture_advisor",
                    "data": {"architectural_advice_raw": "Consider CQRS pattern for scalability.",
                             "primary_finding": "Scalability bottleneck in current design.",
                             "recommendation": "Implement CQRS.",
                             "additional_value": "This will also improve read performance."},
                    "confidence": 0.85}

        async def mock_perf_optimizer_execute(context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "tool_used": "performance_optimizer",
                    "data": {"performance_analysis_raw": "N+1 query problem found in data access layer.",
                             "primary_finding": "N+1 query problem.",
                             "recommendation": "Use eager loading or batch fetching.",
                             "additional_value": "This will reduce database load significantly."},
                    "confidence": 0.92}

        async def mock_failing_tool_execute(tool_name: str, context: JefeContext, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "error", "tool_used": tool_name, "error": f"Mock error from {tool_name}.", "confidence": 0.1}

        # Apply mocks
        if "code_analyzer" in jefe_agent.tools: jefe_agent.tools["code_analyzer"].execute = mock_code_analyzer_execute
        if "debug_assistant" in jefe_agent.tools: jefe_agent.tools["debug_assistant"].execute = mock_debug_assistant_execute
        if "architecture_advisor" in jefe_agent.tools: jefe_agent.tools["architecture_advisor"].execute = mock_arch_advisor_execute
        if "performance_optimizer" in jefe_agent.tools: jefe_agent.tools["performance_optimizer"].execute = mock_perf_optimizer_execute

        print(f"\nJefeAgent Capabilities (loaded {len(jefe_agent.tools)} tools): {jefe_agent.get_capabilities()['tools_available']}")

        # --- Test Scenarios ---

        # Scenario 1: Simple Code Analysis (triggers _handle_simple_assistance)
        print("\n--- Test 1: Simple Code Analysis ---")
        # To force simple, we make task_analyzer result simple and suggest only code_analyzer
        original_task_analyzer_analyze = jefe_agent.task_analyzer.analyze_context
        def mock_analyze_simple_ca(ctx: JefeContext) -> TaskAnalysis:
            return TaskAnalysis(task_type=JefeTaskType.CODING_ASSISTANCE, complexity="simple", immediate_issues=[],
                                suggested_tools=["code_analyzer"], priority="medium", project_context_summary="")
        jefe_agent.task_analyzer.analyze_context = mock_analyze_simple_ca

        context_simple_ca = JefeContext(screen_content="def foo():\n  x = y + z", programming_language="Python")
        response_simple_ca = await jefe_agent.process_realtime_input(context_simple_ca)
        print("Response (Simple CA):")
        print(f"  Content: {response_simple_ca['content']}")
        print(f"  Confidence: {response_simple_ca['confidence']}")
        print(f"  Tools Used: {response_simple_ca['tools_used']}")
        assert "Fix syntax at line 10" in response_simple_ca["content"]
        assert response_simple_ca["tools_used"] == ["code_analyzer"]
        jefe_agent.task_analyzer.analyze_context = original_task_analyzer_analyze # Restore

        # Scenario 2: Architecture Advice (triggers _handle_simple_assistance with arch tool)
        print("\n--- Test 2: Architecture Advice ---")
        def mock_analyze_arch(ctx: JefeContext) -> TaskAnalysis:
            return TaskAnalysis(task_type=JefeTaskType.ARCHITECTURE_ADVICE, complexity="simple", immediate_issues=[],
                                suggested_tools=["architecture_advisor"], priority="high", project_context_summary="")
        jefe_agent.task_analyzer.analyze_context = mock_analyze_arch
        context_arch = JefeContext(audio_transcript="Need advice on scaling my monolith.")
        response_arch = await jefe_agent.process_realtime_input(context_arch)
        print("Response (Architecture):")
        print(f"  Content: {response_arch['content']}")
        print(f"  Confidence: {response_arch['confidence']}")
        print(f"  Tools Used: {response_arch['tools_used']}")
        assert "Implement CQRS" in response_arch["content"]
        assert "architecture_advisor" in response_arch["tools_used"]
        jefe_agent.task_analyzer.analyze_context = original_task_analyzer_analyze

        # Scenario 3: Performance Optimization (triggers _handle_simple_assistance with perf tool)
        print("\n--- Test 3: Performance Optimization ---")
        def mock_analyze_perf(ctx: JefeContext) -> TaskAnalysis:
            return TaskAnalysis(task_type=JefeTaskType.PERFORMANCE_OPTIMIZATION, complexity="simple", immediate_issues=[],
                                suggested_tools=["performance_optimizer"], priority="high", project_context_summary="")
        jefe_agent.task_analyzer.analyze_context = mock_analyze_perf
        context_perf = JefeContext(screen_content="SELECT * FROM users JOIN orders...", audio_transcript="This query is slow.")
        response_perf = await jefe_agent.process_realtime_input(context_perf)
        print("Response (Performance):")
        print(f"  Content: {response_perf['content']}")
        print(f"  Confidence: {response_perf['confidence']}")
        print(f"  Tools Used: {response_perf['tools_used']}")
        assert "Use eager loading" in response_perf["content"]
        assert "performance_optimizer" in response_perf["tools_used"]
        jefe_agent.task_analyzer.analyze_context = original_task_analyzer_analyze

        # Scenario 4: Parallel Tool Usage & Synthesis (triggers _handle_parallel_analysis)
        print("\n--- Test 4: Parallel Tools & Synthesis (Code Analysis + Debugging) ---")
        # Mock task analyzer to suggest multiple tools and moderate complexity
        def mock_analyze_parallel(ctx: JefeContext) -> TaskAnalysis:
            return TaskAnalysis(task_type=JefeTaskType.DEBUGGING_ASSISTANCE, complexity="moderate", immediate_issues=["Error on screen"],
                                suggested_tools=["code_analyzer", "debug_assistant"], priority="high", project_context_summary="")
        jefe_agent.task_analyzer.analyze_context = mock_analyze_parallel

        context_parallel = JefeContext(screen_content="if x == 0: y = 1/x", error_messages=["ZeroDivisionError"])
        response_parallel = await jefe_agent.process_realtime_input(context_parallel)
        print("Response (Parallel CA+Debug):")
        print(f"  Content: {response_parallel['content']}")
        print(f"  Confidence: {response_parallel['confidence']}") # Should be avg of 0.9 and 0.8, weighted by success ratio
        print(f"  Tools Used: {sorted(response_parallel['tools_used'])}") # Sort for consistent assert
        assert "Fix syntax at line 10" in response_parallel["content"] # From CA
        assert "Add null check" in response_parallel["content"]   # From Debug
        assert sorted(response_parallel["tools_used"]) == sorted(["code_analyzer", "debug_assistant"])
        # Expected confidence: ( (0.9+0.8)/2 * 0.7) + (1.0 * 0.3) = (0.85 * 0.7) + 0.3 = 0.595 + 0.3 = 0.895 -> rounded to 0.9
        assert abs(response_parallel['confidence'] - 0.9) < 0.01

        # Sub-Scenario 4b: One tool fails in parallel execution
        print("\n--- Test 4b: Parallel Tools - One Fails ---")
        if "debug_assistant" in jefe_agent.tools: # Ensure tool exists before trying to mock
            jefe_agent.tools["debug_assistant"].execute = lambda context, **kwargs: mock_failing_tool_execute("debug_assistant", context, **kwargs)

        response_parallel_fail = await jefe_agent.process_realtime_input(context_parallel)
        print("Response (Parallel CA success, Debug fail):")
        print(f"  Content: {response_parallel_fail['content']}")
        print(f"  Confidence: {response_parallel_fail['confidence']}")
        print(f"  Tools Used: {sorted(response_parallel_fail['tools_used'])}")
        assert "Fix syntax at line 10" in response_parallel_fail["content"] # CA finding
        assert "Mock error from debug_assistant" in response_parallel_fail["content"] # Error message from failed tool in primary_issue
        assert sorted(response_parallel_fail["tools_used"]) == sorted(["code_analyzer", "debug_assistant"])
        # Expected confidence: successful_confidences = [0.9], avg_successful_confidence = 0.9. success_ratio = 1/2 = 0.5
        # final_confidence = (0.9 * 0.7) + (0.5 * 0.3) = 0.63 + 0.15 = 0.78
        assert abs(response_parallel_fail['confidence'] - 0.78) < 0.01

        # Restore debug_assistant if it was mocked to fail
        if "debug_assistant" in original_tools_execute: jefe_agent.tools["debug_assistant"].execute = mock_debug_assistant_execute
        jefe_agent.task_analyzer.analyze_context = original_task_analyzer_analyze # Restore main analyzer

        # Scenario 5: All Tools Fail (triggers _handle_all_tools_failed via _synthesize_tool_results)
        print("\n--- Test 5: All Tools Fail ---")
        def mock_analyze_multi_fail(ctx: JefeContext) -> TaskAnalysis:
            return TaskAnalysis(task_type=JefeTaskType.GENERAL_ASSISTANCE, complexity="moderate", immediate_issues=[],
                                suggested_tools=["code_analyzer", "performance_optimizer"], priority="medium", project_context_summary="")
        jefe_agent.task_analyzer.analyze_context = mock_analyze_multi_fail

        # Make all relevant tools fail for this test
        if "code_analyzer" in jefe_agent.tools:
            jefe_agent.tools["code_analyzer"].execute = lambda context, **kwargs: mock_failing_tool_execute("code_analyzer", context, **kwargs)
        if "performance_optimizer" in jefe_agent.tools:
            jefe_agent.tools["performance_optimizer"].execute = lambda context, **kwargs: mock_failing_tool_execute("performance_optimizer", context, **kwargs)

        context_all_fail = JefeContext(general_query="This is broken somehow.")
        response_all_fail = await jefe_agent.process_realtime_input(context_all_fail)
        print("Response (All Tools Fail):")
        print(f"  Content: {response_all_fail['content']}")
        print(f"  Status: {response_all_fail['status']}")
        print(f"  Confidence: {response_all_fail['confidence']}")
        print(f"  Tools Used: {sorted(response_all_fail['tools_used'])}")
        assert "Unable to complete analysis due to tool failures" in response_all_fail["identified_issues_summary"] # Check summary field
        assert response_all_fail["status"] == "error"
        assert response_all_fail["confidence"] == 0.1
        assert sorted(response_all_fail["tools_used"]) == sorted(["code_analyzer", "performance_optimizer"])

        jefe_agent.task_analyzer.analyze_context = original_task_analyzer_analyze # Restore

        # Restore original tool execute methods (optional, good practice if tests were longer or part of a suite)
        for tool_key, original_execute in original_tools_execute.items():
            if tool_key in jefe_agent.tools:
                jefe_agent.tools[tool_key].execute = original_execute
        jefe_agent.logger.info("Restored original tool execute methods.")

        # Test Context Compression (minimal, just to ensure it runs without error if history exists)
        print("\n--- Final Test: Context Compression (minimal check) ---")
        if jefe_agent.context_manager.get_full_conversation_history():
            jefe_agent.context_manager.max_history_len_before_compression = 1
            jefe_agent.context_manager.recent_interactions_to_keep = 0
            await jefe_agent.context_manager._compress_old_context()
            print(f"Compressed Memory Summary: {jefe_agent.context_manager.get_compressed_memory_summary()}")
            assert "Mocked LLM response for compression" in jefe_agent.context_manager.get_compressed_memory_summary()
        else:
            print("Skipping active compression test as no interactions were logged to context_manager by these tests.")


    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_jefe_agent_test())
    print("\n--- JefeAgent integration test with mocked tools finished ---")

# TODO:
# - Implement sophisticated tool selection logic in _select_primary_tool_name, _select_parallel_tool_names, etc. (Covered by JefeTaskAnalyzer)
# - Flesh out _update_context_with_findings for complex research. (Still a TODO in main code)
# - Refine interaction between JefeTaskAnalyzer's suggestions and tool selection here. (Partially tested by mocking analyze_context)
# - Add actual tool implementations in the tools directory. (Done for several tools)
