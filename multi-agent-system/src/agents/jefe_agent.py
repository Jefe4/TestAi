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
