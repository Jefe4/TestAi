# src/coordinator/coordinator.py
"""
Main orchestrator for the multi-agent system.

This module defines the Coordinator class, which is responsible for managing
the overall workflow of query processing. It initializes and holds references
to various agents, utilizes a TaskAnalyzer to understand user queries and
generate execution plans, employs a RoutingEngine to select appropriate agents,
and then executes the tasks, potentially in sequence or parallel, managing
data flow between them. It also uses an AdvancedContextManager to trace
execution steps.
"""

from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING # Added Callable, TYPE_CHECKING
import os 
import re 
import asyncio

# Attempt to import project-specific modules.
# The try-except block handles potential ImportError if the module is run directly
# or if the project structure is not correctly recognized by the Python path.
try:
    from ..utils.api_manager import APIManager
    from ..utils.logger import get_logger
    from ..agents.base_agent import BaseAgent
    from .task_analyzer import TaskAnalyzer
    from .routing_engine import RoutingEngine
    from ..agents.deepseek_agent import DeepSeekAgent
    from ..agents.claude_agent import ClaudeAgent
    from ..agents.cursor_agent import CursorAgent 
    from ..agents.windsurf_agent import WindsurfAgent 
    from ..agents.gemini_agent import GeminiAgent
    from ..utils.helpers import get_nested_value
    from ..utils.context_manager import AdvancedContextManager
    from ..utils.jefe_datatypes import JefeContext # Added
    # from ..agents.jefe_agent import JefeAgent # This will be in TYPE_CHECKING
except ImportError:
    # Fallback for environments where relative imports might fail (e.g., direct script execution).
    # This adds the project root to sys.path to help resolve modules.
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Re-attempt imports with the modified path.
    from src.utils.api_manager import APIManager # type: ignore
    from src.utils.logger import get_logger # type: ignore
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.coordinator.task_analyzer import TaskAnalyzer # type: ignore
    from src.coordinator.routing_engine import RoutingEngine # type: ignore
    from src.agents.deepseek_agent import DeepSeekAgent # type: ignore
    from src.agents.claude_agent import ClaudeAgent # type: ignore
    from src.agents.cursor_agent import CursorAgent # type: ignore
    from src.agents.windsurf_agent import WindsurfAgent # type: ignore
    from src.agents.gemini_agent import GeminiAgent # type: ignore
    from src.utils.helpers import get_nested_value # type: ignore
    from src.utils.context_manager import AdvancedContextManager # type: ignore
    from src.utils.jefe_datatypes import JefeContext # type: ignore # Added for fallback
    # from src.agents.jefe_agent import JefeAgent # type: ignore # For fallback, in TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.jefe_agent import JefeAgent


class Coordinator:
    """
    Orchestrates task processing within the multi-agent system.

    The Coordinator is the central component responsible for receiving user queries,
    analyzing them to determine the required steps, selecting the appropriate agents
    for each step, executing the tasks (sequentially or in parallel branches),
    and managing the flow of information between these steps. It maintains a
    registry of available agents and uses helper classes like TaskAnalyzer,
    RoutingEngine, APIManager, and AdvancedContextManager to fulfill its role.
    """

    def __init__(self, agent_config_path: Optional[str] = None):
        """
        Initializes the Coordinator.

        Sets up logging, initializes essential components like APIManager,
        TaskAnalyzer, RoutingEngine, and AdvancedContextManager. It also
        instantiates all configured agents.

        Args:
            agent_config_path: Optional path to the agent configuration YAML file.
                               If None, APIManager will use its default path.
        """
        self.logger = get_logger("Coordinator")
        self.api_manager = APIManager(config_path=agent_config_path)
        self.task_analyzer = TaskAnalyzer() # For analyzing queries and creating plans
        self.routing_engine = RoutingEngine() # For selecting agents based on plans/analysis
        self.context_manager = AdvancedContextManager(api_manager=self.api_manager) # For tracing execution

        self.agents: Dict[str, BaseAgent] = {} # Dictionary to store instantiated agent objects
        self.realtime_processors: Dict[str, Callable] = {} # For specialized real-time input processors like JefeAgent
        self.jefe_agent: Optional['JefeAgent'] = None # Specifically store JefeAgent if registered

        # Mapping of agent type names (from config) to their respective classes
        self.agent_classes: Dict[str, type[BaseAgent]] = {
            "deepseek": DeepSeekAgent, "claude": ClaudeAgent,
            "cursor": CursorAgent, "windsurf": WindsurfAgent,
            "gemini": GeminiAgent,
            # "jefe": JefeAgent, # JefeAgent is registered differently via register_jefe_agent
        }

        self._instantiate_agents() # Load and initialize agents based on configuration

        if self.agents:
            self.logger.info(f"Coordinator initialized successfully with general agents: {', '.join(self.agents.keys())}")
        else:
            self.logger.warning("Coordinator initialized, but no general agents were instantiated. Check configurations.")
        if self.jefe_agent:
            self.logger.info(f"JefeAgent '{self.jefe_agent.get_name()}' is also registered.")


    def register_agent(self, agent_instance: BaseAgent):
        """
        Registers a general agent instance with the Coordinator.

        Args:
            agent_instance: An instance of a class derived from BaseAgent.
        """
        agent_name = agent_instance.get_name()
        if agent_name in self.agents:
            self.logger.warning(f"Agent '{agent_name}' already registered. Overwriting with new instance.")
        self.agents[agent_name] = agent_instance
        self.logger.info(f"Agent '{agent_name}' registered successfully.")

    def _instantiate_agents(self):
        """
        Instantiates agents based on configurations loaded by the APIManager.

        Iterates through service configurations (expected to be agent configurations),
        checks if a corresponding agent class is defined, and if so, creates an
        instance of that agent and registers it.
        """
        self.logger.info("Attempting to instantiate agents based on loaded configurations...")
        agent_configs = self.api_manager.service_configs # Assumes service_configs holds agent configs
        if not agent_configs:
            self.logger.warning("No agent configurations found... Cannot instantiate agents.")
            return
        for agent_key, config_data in agent_configs.items():
            if not isinstance(config_data, dict):
                self.logger.warning(f"Config for agent key '{agent_key}' is not a dict. Skipping.")
                continue
            if agent_key in self.agent_classes:
                AgentClass = self.agent_classes[agent_key]
                agent_name = config_data.get("name", agent_key)
                if agent_key != "gemini" and not config_data.get("api_key"):
                    self.logger.warning(f"API key missing for agent '{agent_name}'. Skipping.")
                    continue
                try:
                    agent_instance = AgentClass(agent_name=agent_name, api_manager=self.api_manager, config=config_data)
                    self.register_agent(agent_instance)
                except Exception as e:
                    self.logger.error(f"Failed to instantiate agent '{agent_name}': {e}", exc_info=True)
            else:
                self.logger.warning(f"No agent class for config key: '{agent_key}'. Skipping.")
        self.logger.info(f"Agent instantiation completed. Agents: {list(self.agents.keys())}")

    def _resolve_input_value(
        self,
        mapping_config: Dict[str, Any],
        initial_query_input: str,
        execution_context: Dict[str, Any],
        current_step_index: int,
        full_plan_details: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """
        Resolves the actual value for a specific input field of an agent's query
        based on its mapping configuration.

        The input value can be:
        1. A direct literal value (`"value": "some_string"`).
        2. Sourced from the original user query (`"source": "original_query"`).
        3. Sourced from the output of a previous step (`"source": "ref:step_id.field_path"`
           or `"source": "ref:previous_step.content"`).
        4. Constructed from a template string with placeholders for original query or previous step outputs
           (`"template": "Combine {original_query} with {ref:step1_output.some_field}"`).

        Args:
            mapping_config: The configuration for this specific input field,
                            e.g., `{"source": "original_query"}` or
                            `{"value": "static text"}` or
                            `{"source": "ref:previous_step.content"}` or
                            `{"template": "Input: {original_query}"}`.
            initial_query_input: The original query string submitted by the user.
            execution_context: A dictionary holding outputs of all previously
                               executed steps, keyed by their `output_id`.
            current_step_index: The index of the current step being processed within
                                `full_plan_details`. Used for "previous_step" refs.
            full_plan_details: The list of all step definitions in the current scope
                               (e.g., a branch of a parallel plan, or the main sequential plan).
                               Used to find `output_id` of the previous step.

        Returns:
            The resolved value for the input, or None if resolution fails or
            a required source is unavailable.
        """
        # 1. Direct literal value
        if "value" in mapping_config:
            return mapping_config["value"]

        source_uri = mapping_config.get("source")

        # 2. Template-based value
        # Placed before direct source resolution to allow templates to use "original_query" or refs.
        if "template" in mapping_config:
            template_str = mapping_config["template"]
            # This replacer function is called for each match of the regex below.
            def replacer(match: re.Match) -> str:
                placeholder_type = match.group(1) # 'original_query' or 'ref'
                reference_path = match.group(2)   # The actual path, e.g., 'step_id.field' or empty if original_query

                if placeholder_type == "original_query":
                    return initial_query_input

                # For "ref:..." type placeholders
                if '.' not in reference_path:
                    self.logger.warning(f"Template reference '{reference_path}' invalid format. Expected 'step_id.field_path'. Placeholder used.")
                    return f"[INVALID_REF_FORMAT: {reference_path}]"

                target_step_id, field_path = reference_path.split('.', 1)
                response_object = execution_context.get(target_step_id)

                if response_object is None or not isinstance(response_object, dict) or response_object.get("status") != "success":
                    self.logger.warning(f"Template reference step '{target_step_id}' not found in execution context or failed. Placeholder used.")
                    return f"[DATA_NOT_FOUND_FOR_{target_step_id}]"

                _NOT_FOUND_IN_TEMPLATE = object()
                value = get_nested_value(response_object, field_path, default=_NOT_FOUND_IN_TEMPLATE)

                if value is _NOT_FOUND_IN_TEMPLATE:
                    self.logger.warning(f"Template reference path '{field_path}' in step '{target_step_id}' output not found. Placeholder used.")
                    return f"[FIELD_{field_path.replace('.', '_')}_NOT_FOUND_IN_{target_step_id}]"
                return str(value)

            try:
                # Regex to find {original_query} or {ref:step_id.field.subfield}
                return re.sub(r"\{(original_query|ref:([^}]+))\}", replacer, template_str)
            except Exception as e:
                self.logger.error(f"Error processing template string '{template_str}': {e}", exc_info=True)
                return None

        # 3. Direct source URI (original_query or reference)
        if not source_uri:
            self.logger.warning(f"Input mapping {mapping_config} missing 'source' or 'value'. Returning None.")
            return None
        if source_uri == "original_query": return initial_query_input
        if source_uri.startswith("ref:"):
            ref_path = source_uri.split("ref:", 1)[1]
            target_step_output_id, field_path_str = "", "" # Initialize
            
            # If source_uri is "original_query"
            if source_uri == "original_query":
                return initial_query_input

            # If source_uri starts with "ref:" (reference to a previous step's output)
            if source_uri.startswith("ref:"):
                ref_path = source_uri.split("ref:", 1)[1]
                target_step_output_id: Optional[str] = None
                field_path_str: str = ""

                # Handle "previous_step.content" specifically
                if ref_path == "previous_step.content":
                    if current_step_index == 0:
                        self.logger.error("Cannot use 'ref:previous_step.content' for the first step of a sequence.")
                        return None
                    # Get the output_id of the immediately preceding step in the current plan/branch
                    prev_step_def = full_plan_details[current_step_index - 1]
                    target_step_output_id = prev_step_def.get("output_id")
                    if not target_step_output_id:
                        self.logger.error(f"Previous step (index {current_step_index - 1}) is missing an 'output_id'. Cannot reference 'previous_step.content'.")
                        return None
                    field_path_str = "content" # Default field for "previous_step.content"
                else:
                    # Handle "ref:step_id.field.subfield"
                    if '.' not in ref_path:
                        self.logger.error(f"Invalid reference path '{ref_path}'. Must be 'step_id.field_path' or 'previous_step.content'.")
                        return None
                    target_step_output_id, field_path_str = ref_path.split('.', 1)

                # Retrieve the referenced step's output from the execution context
                target_response_object = execution_context.get(target_step_output_id)
                if target_response_object is None:
                    self.logger.error(f"Referenced step output '{target_step_output_id}' not found in execution_context.")
                    return None

                # Ensure the referenced output indicates success and is a dictionary
                if not isinstance(target_response_object, dict) or target_response_object.get("status") != "success":
                    self.logger.error(f"Referenced step '{target_step_output_id}' either failed or its output is not a dictionary: {str(target_response_object)[:200]}")
                    return None

                # Use helper to get potentially nested value from the response object
                _NOT_FOUND_VALUE = object()
                resolved_data = get_nested_value(target_response_object, field_path_str, default=_NOT_FOUND_VALUE)

                if resolved_data is _NOT_FOUND_VALUE:
                    self.logger.error(f"Field path '{field_path_str}' not found in the output of step '{target_step_output_id}'.")
                    return None
                return resolved_data
            
        # If mapping_config is not recognized or source_uri is invalid after template check
        self.logger.warning(f"Unknown input mapping type or invalid source URI in {mapping_config} (and not a template). Returning None.")
        return None


    async def _execute_branch_sequentially(
        self,
        branch_step_templates: List[Dict[str, Any]],
        initial_branch_input: str,
        branch_id_for_logging: str,
        global_query_data_overrides: Optional[Dict[str, Any]],
        main_execution_context: Dict[str, Any],
        context_manager: AdvancedContextManager
    ) -> Optional[Dict[str, Any]]:
        """
        Executes a sequence of agent tasks defined within a branch of a parallel plan.

        This method iterates through the step templates in a branch, resolves
        inputs for each step (potentially using outputs from previous steps within
        the same branch or from the main execution context before the parallel block),
        calls the specified agent, and collects the results. If any step fails,
        the branch execution stops and returns the error.

        Args:
            branch_step_templates: A list of step definition dictionaries for this branch.
            initial_branch_input: The initial input that can be referenced by steps
                                  in this branch as "original_query". This is typically
                                  the main query input to the Coordinator.
            branch_id_for_logging: A string identifier for this branch, used for logging.
            global_query_data_overrides: Optional dictionary of overrides that might apply
                                         to all agents in this branch, unless overridden by
                                         step-specific `agent_config_overrides`.
            main_execution_context: The execution context from *before* the parallel block started.
                                    Used for resolving `ref:step_id.field` that point to outputs
                                    generated outside this specific branch.
            context_manager: The AdvancedContextManager instance for logging trace events.

        Returns:
            The response from the last successfully executed agent in the branch,
            or an error dictionary if any step fails. Returns None if the branch
            is empty or an unexpected issue occurs.
        """
        self.logger.info(f"Executing branch '{branch_id_for_logging}' with {len(branch_step_templates)} steps.")
        branch_internal_context: Dict[str, Any] = {} # Context for outputs *within* this branch
        last_branch_response: Optional[Dict[str, Any]] = None

        for i, step_def in enumerate(branch_step_templates):
            agent_name = step_def["agent_name"]
            if agent_name not in self.agents:
                self.logger.error(f"Agent '{agent_name}' in branch '{branch_id_for_logging}' not available. Branch execution fails.")
                return {"status": "error", "message": f"Agent '{agent_name}' for branch '{branch_id_for_logging}' not found."}

            current_agent = self.agents[agent_name]
            current_agent_inputs: Dict[str, Any] = {}
            input_mapping_config = step_def.get("input_mapping", {})
            
            # Resolve each input field for the current agent step
            for input_key, mapping_config_item in input_mapping_config.items():
                # Determine the correct context for input resolution:
                # - If source is "ref:previous_step.*", use `branch_internal_context`.
                # - Otherwise (e.g. "original_query" or "ref:some_global_step_id.*"), use `main_execution_context`.
                #   (Note: `_resolve_input_value` itself uses `initial_branch_input` for "original_query")
                context_for_resolution = main_execution_context
                if mapping_config_item.get("source", "").startswith("ref:previous_step"):
                    context_for_resolution = branch_internal_context

                resolved_value = self._resolve_input_value(
                    mapping_config_item,
                    initial_branch_input, # This is the input to the whole parallel block / initial query
                    context_for_resolution,
                    i, # current_step_index *within the branch*
                    branch_step_templates # full_plan_details *for this branch*
                )

                if resolved_value is None and mapping_config_item.get("required", True):
                    error_msg = (f"Critical input '{input_key}' for agent '{agent_name}' in branch "
                                 f"'{branch_id_for_logging}', step {i+1}, could not be resolved.")
                    self.logger.error(error_msg)
                    return {"status": "error", "message": error_msg, "agent_name": agent_name}
                current_agent_inputs[input_key] = resolved_value
            
            agent_query_data = current_agent_inputs.copy()

            # Handle Gemini-specific input adaptation (prompt vs prompt_parts)
            gemini_agent_class = self.agent_classes.get("gemini")
            is_gemini_instance = gemini_agent_class and isinstance(current_agent, gemini_agent_class)
            if is_gemini_instance: # Adapt for Gemini
                if "prompt" in agent_query_data and agent_query_data["prompt"] is not None:
                    agent_query_data["prompt_parts"] = [str(agent_query_data.pop("prompt"))]
                elif "prompt_parts" not in agent_query_data:
                     self.logger.warning(f"Gemini agent {current_agent.get_name()} in branch {branch_id_for_logging} called without prompt/prompt_parts.")
            elif "prompt_parts" in agent_query_data: # Non-Gemini with prompt_parts
                 self.logger.warning(f"Non-Gemini agent {current_agent.get_name()} in branch {branch_id_for_logging} received 'prompt_parts'.")

            effective_configs = {}
            if global_query_data_overrides: effective_configs.update(global_query_data_overrides.copy())
            if step_def.get("agent_config_overrides"): effective_configs.update(step_def["agent_config_overrides"])
            for key in current_agent_inputs.keys(): effective_configs.pop(key, None) # Inputs take precedence
            if "system_prompt" in effective_configs: agent_query_data["system_prompt"] = effective_configs.pop("system_prompt")
            # Note: system_prompt from analysis_result is not easily applied here unless passed in.
            # TaskAnalyzer should put system_prompt into agent_config_overrides if step-specific.
            agent_query_data.update(effective_configs)

            self.logger.debug(f"Branch '{branch_id_for_logging}' step {i+1} ({agent_name}) query data: {str(agent_query_data)[:200]}...")
            context_manager.add_trace_event(
                "branch_agent_call_start",
                {"branch_id": branch_id_for_logging, "step_index": i, "agent_name": agent_name, "query_data": agent_query_data}
            )
            try:
                response = await current_agent.process_query(agent_query_data)
            except Exception as e:
                error_msg = f"Exception during agent call in branch '{branch_id_for_logging}', step {i+1} ({agent_name}): {e}"
                self.logger.error(error_msg, exc_info=True)
                context_manager.add_trace_event(
                    "branch_agent_call_error",
                    {"branch_id": branch_id_for_logging, "step_index": i, "agent_name": agent_name, "error": str(e)}
                )
                return {"status": "error", "message": error_msg, "agent_name": agent_name}

            # Store response in the branch's internal context if an output_id is specified
            if step_def.get("output_id"):
                branch_internal_context[step_def["output_id"]] = response

            last_branch_response = response # Keep track of the last response in the branch
            context_manager.add_trace_event(
                "branch_agent_call_end",
                {"branch_id": branch_id_for_logging, "step_index": i, "agent_name": agent_name, "response": response}
            )

            # If agent call was not successful, stop processing this branch
            if response.get("status") != "success":
                self.logger.warning(f"Agent '{agent_name}' in branch '{branch_id_for_logging}' (step {i+1}) returned error: {response.get('message')}")
                return response # Propagate the error response
            
            # If not the last step in the branch and the response content is None (needed for chaining), it's an error.
            if response.get("content") is None and i < len(branch_step_templates) - 1:
                error_msg = f"Agent '{agent_name}' in branch '{branch_id_for_logging}' (step {i+1}) produced no 'content' for chaining to the next step."
                self.logger.error(error_msg)
                # Note: No trace event for this specific internal error, could be added.
                return {"status": "error", "message": error_msg, "agent_name": agent_name}

            # The `current_input_content` variable previously here was for implicit chaining.
            # Explicit input_mapping via `_resolve_input_value` now handles chaining.

        self.logger.info(f"Branch '{branch_id_for_logging}' completed. Final response: {str(last_branch_response)[:100]}...")
        return last_branch_response


    async def process_query(self, query: str, query_data_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a user query by analyzing it, routing it to appropriate agents
        based on an execution plan (sequential or parallel), and returning the final result.

        The method orchestrates the following high-level steps:
        1. Clears any previous execution trace and logs the new query.
        2. Analyzes the query using `TaskAnalyzer` to get an execution plan and other insights.
        3. Uses `RoutingEngine` to select agent instances based on the plan or analysis.
        4. Executes the plan:
            - If the plan is a parallel block: Executes branches concurrently using `_execute_branch_sequentially`.
            - If the plan is sequential: Executes steps one by one, resolving inputs and passing outputs.
            - If no plan but a single agent is suggested: Executes that single agent.
            - If multiple agents are suggested without a plan (legacy): Executes them sequentially with basic chaining.
        5. Logs trace events at each significant stage using `AdvancedContextManager`.
        6. Returns the final response from the last executed agent or aggregated results from parallel execution.

        Args:
            query: The user's query string.
            query_data_overrides: Optional dictionary of parameters that can override
                                  default agent configurations or query data for this specific query.
                                  Example: `{"temperature": 0.8, "max_tokens": 500}`.

        Returns:
            A dictionary containing the status of the operation and the response content
            or an error message.
        """
        self.logger.info(f"Coordinator received query: '{query[:100]}...'")
        self.context_manager.clear_trace() # Start with a fresh trace for each new query
        self.context_manager.add_trace_event("query_received", {"query": query, "query_data_overrides": query_data_overrides if query_data_overrides else {}})

        if not self.agents: # Check if any agents are loaded
            self.logger.error("No agents registered. Cannot process query.")
            return {"status": "error", "message": "No agents available."}

        # Analyze the query to get an execution plan and other metadata
        analysis_result = self.task_analyzer.analyze_query(query, self.agents, context_trace=self.context_manager.get_full_trace())
        self.context_manager.add_trace_event("task_analysis_complete", {"analysis_result": analysis_result})
        self.logger.debug(f"Task analysis result: {analysis_result}")
        
        # Select agent instances based on the analysis and routing logic
        selected_agent_instances_from_router = self.routing_engine.select_agents(analysis_result, self.agents)
        self.context_manager.add_trace_event(
            "agent_routing_complete",
            {"selected_agent_names": [agent.get_name() for agent in selected_agent_instances_from_router],
             "num_selected": len(selected_agent_instances_from_router)}
        )
        
        if not selected_agent_instances_from_router: # If router finds no suitable agents
            self.logger.warning("RoutingEngine selected no agents based on the analysis.")
            self.context_manager.add_trace_event("routing_error", {"message": "Router selected no agents"})
            return {"status": "error", "message": "No suitable agent found for the query."}

        execution_plan_details = analysis_result.get("execution_plan") # The plan from TaskAnalyzer
        initial_query_input = str(analysis_result.get("processed_query_for_agent", query)) # Query text for agents
        execution_context: Dict[str, Any] = {} # Stores outputs of steps for later reference (input chaining)
        final_response: Optional[Dict[str, Any]] = None # The final result to be returned

        # --- Parallel Block Execution ---
        # Check if the execution plan consists of a single top-level parallel block
        if execution_plan_details and \
           len(execution_plan_details) == 1 and \
           isinstance(execution_plan_details[0], dict) and \
           execution_plan_details[0].get("type") == "parallel_block":

            parallel_block_def = execution_plan_details[0]
            self.logger.info(f"Executing parallel block: {parallel_block_def.get('task_description', 'N/A')}")
            self.context_manager.add_trace_event("parallel_block_start", {"block_definition": parallel_block_def})
            
            branch_templates = parallel_block_def.get("branches", [])
            branch_coroutines = [] # List to hold coroutines for each branch execution

            # Prepare coroutines for each branch
            for idx, branch_steps in enumerate(branch_templates):
                if not branch_steps: # Skip empty branches
                    self.logger.warning(f"Skipping empty branch {idx} in parallel block.")
                    continue

                # Each branch is executed sequentially using `_execute_branch_sequentially`.
                # `main_execution_context` is passed for refs outside the branch.
                # `initial_query_input` is the overall query input.
                branch_coroutines.append(
                    self._execute_branch_sequentially(
                        branch_steps,
                        initial_query_input,
                        f"branch_{idx}", 
                        query_data_overrides,
                        execution_context.copy(), # Pass a copy of context from *before* this parallel block
                        self.context_manager
                    )
                )
            
            if not branch_coroutines:
                self.logger.warning("Parallel block defined but has no executable branches.")
                self.context_manager.add_trace_event("parallel_block_error", {"message": "No executable branches"})
                return {"status": "error", "message": "Parallel block has no executable branches."}

            # Execute all branches concurrently
            branch_execution_results = await asyncio.gather(*branch_coroutines, return_exceptions=True)
            
            # Aggregate results from branches
            aggregated_results_data: Dict[str, Any] = {}
            all_branches_succeeded = True
            for idx, result in enumerate(branch_execution_results):
                branch_key = f"branch_{idx}_result"
                if isinstance(result, Exception):
                    self.logger.error(f"Branch {idx} execution raised an exception: {result}", exc_info=result)
                    aggregated_results_data[branch_key] = {"status": "error", "message": f"Branch execution raised: {str(result)}", "details": str(result)}
                    all_branches_succeeded = False
                elif result is None or result.get("status") != "success":
                    self.logger.warning(f"Branch {idx} failed or returned no data: {result}")
                    aggregated_results_data[branch_key] = result or {"status": "error", "message": "Branch failed or returned no data."}
                    all_branches_succeeded = False
                else:
                    aggregated_results_data[branch_key] = result
            
            # Store the aggregated result of the parallel block in the main execution context
            pb_output_id = parallel_block_def.get("output_id", "parallel_block_output")
            current_step_result = {
                "status": "success" if all_branches_succeeded else "partial_success", 
                "type": "parallel_block_result",
                "output_id": pb_output_id,
                "aggregated_results": aggregated_results_data
            }
            execution_context[pb_output_id] = current_step_result 
            final_response = current_step_result # This becomes the final response if it's the only top-level step
            self.context_manager.add_trace_event("parallel_block_end", {"output_id": pb_output_id, "result_summary": current_step_result})

        # --- Sequential Plan Execution ---
        # This block handles plans that are a list of steps (not a single parallel block at the root).
        elif execution_plan_details:
            self.logger.info(f"Starting sequential execution for plan: {[s.get('agent_name', 'UnknownAgent') for s in execution_plan_details]}")
            # Note: `selected_agent_instances_from_router` should ideally be used here to get actual agent instances
            # if the plan only contains names. However, the current RoutingEngine logic for sequential plans
            # already returns the list of agent instances in order.

            # `execution_context` is used here directly (instead of a temp_execution_context)
            # because this is the main sequential flow of the query.
            # `initial_query_input` is the input for the first step if it uses "original_query".

            for step_idx, step_detail in enumerate(execution_plan_details):
                agent_name_from_plan = step_detail.get("agent_name")
                if not agent_name_from_plan or agent_name_from_plan not in self.agents:
                    error_msg = f"Agent '{agent_name_from_plan}' in sequential plan (step {step_idx+1}) not found or not available."
                    self.logger.error(error_msg)
                    self.context_manager.add_trace_event("sequential_plan_error", {"message": error_msg, "step": step_idx})
                    return {"status": "error", "message": error_msg}

                current_agent = self.agents[agent_name_from_plan]

                # Resolve inputs for the current step
                agent_inputs: Dict[str, Any] = {}
                for input_key, map_config in step_detail.get("input_mapping", {}).items():
                    resolved_value = self._resolve_input_value(
                        map_config,
                        initial_query_input,
                        execution_context, # Use the main execution_context for sequential plan
                        step_idx,
                        execution_plan_details
                    )
                    if resolved_value is None and map_config.get("required", True):
                        error_msg = f"Failed to resolve required input '{input_key}' for agent '{agent_name_from_plan}' in sequential plan (step {step_idx+1})."
                        self.logger.error(error_msg)
                        self.context_manager.add_trace_event("sequential_plan_error", {"message": error_msg, "step": step_idx, "input_key": input_key})
                        return {"status": "error", "message": error_msg}
                    agent_inputs[input_key] = resolved_value

                agent_query_data = agent_inputs.copy()

                # Adapt for Gemini Agent if necessary (prompt vs prompt_parts)
                gemini_agent_class = self.agent_classes.get("gemini")
                is_gemini_instance = gemini_agent_class and isinstance(current_agent, gemini_agent_class)
                if is_gemini_instance:
                    if "prompt" in agent_query_data and agent_query_data["prompt"] is not None:
                        agent_query_data["prompt_parts"] = [str(agent_query_data.pop("prompt"))]
                    elif "prompt_parts" not in agent_query_data:
                        self.logger.warning(f"Gemini agent '{current_agent.get_name()}' in sequential plan (step {step_idx+1}) called without prompt/prompt_parts.")
                elif "prompt_parts" in agent_query_data: # Non-Gemini agent with prompt_parts
                     self.logger.warning(f"Non-Gemini agent '{current_agent.get_name()}' in sequential plan (step {step_idx+1}) received 'prompt_parts'.")

                # Apply global and step-specific overrides
                effective_configs = {}
                if query_data_overrides: effective_configs.update(query_data_overrides.copy())
                if step_detail.get("agent_config_overrides"): effective_configs.update(step_detail["agent_config_overrides"])
                for key_to_remove in agent_inputs.keys(): effective_configs.pop(key_to_remove, None) # Inputs take precedence

                # Handle system_prompt specifically: overrides > step_config > analysis_result (for first step)
                if "system_prompt" in effective_configs:
                    agent_query_data["system_prompt"] = effective_configs.pop("system_prompt")
                elif step_idx == 0 and analysis_result.get("system_prompt") and "system_prompt" not in agent_query_data:
                    # Apply system_prompt from TaskAnalyzer only to the first agent in a sequential plan
                    # if not already set by overrides or input_mapping.
                    agent_query_data["system_prompt"] = analysis_result.get("system_prompt")
                agent_query_data.update(effective_configs) # Apply remaining overrides

                self.logger.debug(f"Sequential plan step {step_idx+1} ({agent_name_from_plan}) query data: {str(agent_query_data)[:200]}...")
                self.context_manager.add_trace_event(
                    "sequential_agent_call_start",
                    {"step_index": step_idx, "agent_name": agent_name_from_plan, "query_data": agent_query_data}
                )
                try:
                    step_response = await current_agent.process_query(agent_query_data)
                except Exception as e:
                    error_msg = f"Exception in sequential plan step {step_idx+1} ({agent_name_from_plan}): {e}"
                    self.logger.error(error_msg, exc_info=True)
                    self.context_manager.add_trace_event("sequential_agent_call_error", {"step": step_idx, "agent_name": agent_name_from_plan, "error": str(e)})
                    return {"status": "error", "message": error_msg, "agent_name": agent_name_from_plan}

                # Store step output in execution_context if output_id is defined
                if step_detail.get("output_id"):
                    execution_context[step_detail["output_id"]] = step_response

                final_response = step_response # The response of the last agent in sequence is the final one
                self.context_manager.add_trace_event(
                    "sequential_agent_call_end",
                    {"step_index": step_idx, "agent_name": agent_name_from_plan, "response": step_response}
                )

                if step_response.get("status") != "success":
                    self.logger.warning(f"Agent '{agent_name_from_plan}' in sequential plan (step {step_idx+1}) returned error: {step_response.get('message')}")
                    return step_response # Propagate error and stop sequence

                # If not the last step, and content is None, this might be an issue for chaining.
                if step_response.get("content") is None and step_idx < len(execution_plan_details) - 1:
                     self.logger.warning(f"Agent '{agent_name_from_plan}' in sequential plan (step {step_idx+1}) produced no 'content'. This might affect chaining.")
                     # Depending on strictness, could return error here. For now, allow continuation.
            # End of sequential plan loop
        
        # --- Single Agent Execution (No specific plan, router suggested one agent) ---
        elif len(selected_agent_instances_from_router) == 1:
            primary_agent = selected_agent_instances_from_router[0]
            self.logger.info(f"Executing with single suggested agent: {primary_agent.get_name()}")

            agent_query_data: Dict[str, Any] = {}
            processed_prompt = analysis_result.get("processed_query_for_agent", query) # Use processed query

            # Adapt for Gemini Agent if necessary
            gemini_agent_class = self.agent_classes.get("gemini")
            is_gemini_instance = gemini_agent_class and isinstance(primary_agent, gemini_agent_class)
            if is_gemini_instance:
                agent_query_data["prompt_parts"] = [processed_prompt]
            else:
                agent_query_data["prompt"] = processed_prompt

            # Apply system prompt from analysis if available
            if analysis_result.get("system_prompt"):
                agent_query_data["system_prompt"] = analysis_result.get("system_prompt")

            # Apply global query_data_overrides
            if query_data_overrides:
                temp_overrides = query_data_overrides.copy()
                # Ensure prompt/prompt_parts from overrides don't conflict if already set
                if is_gemini_instance:
                    if "prompt_parts" in temp_overrides: agent_query_data["prompt_parts"] = temp_overrides.pop("prompt_parts")
                    if "prompt" in temp_overrides and "prompt_parts" in agent_query_data: temp_overrides.pop("prompt", None) # Avoid conflict
                else: 
                    if "prompt" in temp_overrides: agent_query_data["prompt"] = temp_overrides.pop("prompt")
                    if "prompt_parts" in temp_overrides: temp_overrides.pop("prompt_parts", None) # Avoid conflict
                agent_query_data.update(temp_overrides)

            self.logger.debug(f"Single agent query data for '{primary_agent.get_name()}': {str(agent_query_data)[:200]}...")
            self.context_manager.add_trace_event(
                "single_agent_call_start",
                {"agent_name": primary_agent.get_name(), "query_data": agent_query_data}
            )
            try:
                final_response = await primary_agent.process_query(agent_query_data)
                self.context_manager.add_trace_event(
                    "single_agent_call_end",
                    {"agent_name": primary_agent.get_name(), "response": final_response}
                )
            except Exception as e:
                self.logger.error(f"Error during single agent call with '{primary_agent.get_name()}': {e}", exc_info=True)
                self.context_manager.add_trace_event(
                    "single_agent_call_error",
                    {"agent_name": primary_agent.get_name(), "error": str(e)}
                )
                return {"status": "error", "message": f"Failed query with agent '{primary_agent.get_name()}': {str(e)}"}

        # --- Legacy Sequential Fallback (Multiple agents suggested by router, but no explicit plan from TaskAnalyzer) ---
        else:
            self.logger.warning(
                f"Executing legacy simple sequential logic for {len(selected_agent_instances_from_router)} agents. "
                "This typically means TaskAnalyzer did not produce a specific plan, and RoutingEngine suggested multiple agents."
            )
            current_input_content = initial_query_input # Start with the initial (possibly processed) query
            legacy_final_response: Dict[str, Any] = {} # To store the final response of this sequence

            for i, agent_instance in enumerate(selected_agent_instances_from_router):
                self.logger.info(f"Legacy sequential step {i+1}: Agent - {agent_instance.get_name()}")

                # Prepare query data: use content from previous agent as prompt for current.
                # This is a simple form of chaining.
                agent_query_data = {"prompt": current_input_content}
                if query_data_overrides: # Apply overrides, ensuring not to overwrite the chained prompt.
                    temp_overrides = query_data_overrides.copy()
                    temp_overrides.pop("prompt", None); temp_overrides.pop("prompt_parts", None) # Remove conflicting keys
                    agent_query_data.update(temp_overrides)

                self.context_manager.add_trace_event(
                    "legacy_sequential_agent_call_start",
                    {"step_index": i, "agent_name": agent_instance.get_name(), "query_data": agent_query_data}
                )
                try:
                    response = await agent_instance.process_query(agent_query_data)
                    self.context_manager.add_trace_event(
                        "legacy_sequential_agent_call_end",
                        {"step_index": i, "agent_name": agent_instance.get_name(), "response": response}
                    )
                    legacy_final_response = response # Update final response with the latest one

                    if response.get("status") != "success":
                        self.logger.warning(f"Agent '{agent_instance.get_name()}' in legacy sequence failed. Stopping sequence.")
                        return response # Return error and stop

                    new_content = response.get("content")
                    if new_content is None and i < len(selected_agent_instances_from_router) - 1:
                        self.logger.warning(f"Agent '{agent_instance.get_name()}' in legacy sequence produced no content for chaining. Stopping.")
                        return {"status": "error", "message": f"Agent '{agent_instance.get_name()}' produced no content for chaining."}
                    current_input_content = str(new_content if new_content is not None else "") # Prepare for next iteration

                except Exception as e:
                    self.logger.error(f"Error in legacy sequential execution with agent '{agent_instance.get_name()}': {e}", exc_info=True)
                    self.context_manager.add_trace_event(
                        "legacy_sequential_agent_call_error",
                        {"step_index": i, "agent_name": agent_instance.get_name(), "error": str(e)}
                    )
                    return {"status": "error", "message": f"Error in legacy sequence with agent '{agent_instance.get_name()}': {str(e)}"}
            final_response = legacy_final_response # The result of the last agent in the chain
        
        # Log the final response event and return
        self.context_manager.add_trace_event("final_response_generated", {"final_response": final_response if final_response is not None else {}})
        return final_response if final_response is not None else {"status": "error", "message": "Processing resulted in no final response."}

    # --- New methods for JefeAgent and real-time processing ---

    def register_realtime_processor(self, name: str, processor_func: Callable):
        """
        Registers a function capable of handling real-time context processing.
        Currently primarily used for JefeAgent.

        Args:
            name: A name for the real-time processor (e.g., "jefe_realtime").
            processor_func: The async function to call for processing real-time input.
                            Expected signature: `async def func(jefe_context: JefeContext) -> Dict[str, Any]`
        """
        if name in self.realtime_processors:
            self.logger.warning(f"Real-time processor '{name}' already registered. Overwriting.")
        self.realtime_processors[name] = processor_func
        self.logger.info(f"Real-time processor '{name}' registered.")

    def register_jefe_agent(self, jefe_agent_instance: 'JefeAgent'):
        """
        Registers a specific JefeAgent instance.

        This also registers its `process_realtime_input` method as the default
        real-time processor if no other is set.

        Args:
            jefe_agent_instance: An instance of JefeAgent.
        """
        self.register_agent(jefe_agent_instance) # Register as a general agent too
        self.jefe_agent = jefe_agent_instance
        # Register JefeAgent's main real-time processing method
        if "jefe_realtime_processor" not in self.realtime_processors: # Avoid overwriting if manually set
            self.register_realtime_processor(
                "jefe_realtime_processor",
                jefe_agent_instance.process_realtime_input
            )
        self.logger.info(f"JefeAgent '{jefe_agent_instance.get_name()}' specifically registered. Its real-time processor is now active.")


    async def process_realtime_input(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes real-time contextual input using the registered JefeAgent.

        Args:
            context_data: A dictionary containing data to construct a JefeContext.
                          Expected keys match JefeContext fields (e.g., "screen_content",
                          "audio_transcript", "current_ide", "programming_language", etc.).

        Returns:
            A dictionary containing the response from the JefeAgent, or an error message.
        """
        self.logger.info(f"Received real-time input for processing: {str(context_data)[:200]}...")
        self.context_manager.clear_trace() # Start fresh trace for this input
        self.context_manager.add_trace_event("realtime_input_received", {"context_data": context_data})

        if not self.jefe_agent:
            self.logger.error("JefeAgent not registered. Cannot process real-time input.")
            self.context_manager.add_trace_event("realtime_processing_error", {"message": "JefeAgent not registered."})
            return {"status": "error", "message": "JefeAgent not available."}

        try:
            # Construct JefeContext, providing defaults for optional fields if not in context_data
            jefe_context = JefeContext(
                screen_content=context_data.get("screen_content", ""),
                audio_transcript=context_data.get("audio_transcript", ""),
                current_ide=context_data.get("current_ide"),
                programming_language=context_data.get("programming_language"),
                project_type=context_data.get("project_type"),
                error_messages=context_data.get("error_messages", []),
                previous_suggestions=context_data.get("previous_suggestions", [])
            )

            # Use the registered real-time processor, which should be JefeAgent's method
            processor_func = self.realtime_processors.get("jefe_realtime_processor")
            if not processor_func: # Should not happen if register_jefe_agent was called
                 self.logger.error("No 'jefe_realtime_processor' registered despite JefeAgent being present.")
                 return {"status": "error", "message": "Jefe real-time processor misconfiguration."}

            response = await processor_func(jefe_context)
            self.context_manager.add_trace_event("realtime_processing_complete", {"response": response})
            return response
        except Exception as e:
            self.logger.error(f"Error during real-time input processing with JefeAgent: {e}", exc_info=True)
            self.context_manager.add_trace_event("realtime_processing_exception", {"error": str(e)})
            return {"status": "error", "message": f"Exception during real-time processing: {str(e)}"}

    def should_use_realtime_processing(self, input_data: Dict[str, Any]) -> bool:
        """
        Determines if the input data suggests a real-time processing flow (JefeAgent)
        versus a standard query processing flow.

        Args:
            input_data: The input data dictionary.

        Returns:
            True if real-time processing should be used, False otherwise.
        """
        # Check for specific keys that indicate rich real-time context
        if any(key in input_data for key in ['screen_content', 'audio_transcript', 'current_ide']):
            return True
        # Check for an explicit source indicator
        if input_data.get('source') == 'realtime':
            return True
        # If it only contains 'query', it's likely a standard text query
        if 'query' in input_data and len(input_data) == 1:
            return False
        if 'query' in input_data and 'overrides' in input_data and len(input_data) == 2:
             return False

        # Default: if not clearly a standard query, and JefeAgent is present, consider real-time.
        # This heuristic might need refinement.
        return bool(self.jefe_agent) if not input_data.get("query") else False


    async def unified_process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified entry point for processing any input.
        Routes to either real-time processing (JefeAgent) or standard query processing.

        Args:
            input_data: A dictionary that can either contain keys for `JefeContext`
                        (e.g., 'screen_content', 'audio_transcript', 'source': 'realtime')
                        OR keys for standard query processing ('query', 'overrides').

        Returns:
            The processing result as a dictionary.
        """
        if self.should_use_realtime_processing(input_data):
            self.logger.info("Routing to real-time processing (JefeAgent).")
            # Ensure 'source' key is removed if it was only for routing,
            # as process_realtime_input expects JefeContext fields.
            # However, context_data for JefeContext.from_dict would handle extra keys.
            return await self.process_realtime_input(input_data)
        else:
            self.logger.info("Routing to standard query processing.")
            query = input_data.get("query", "")
            if not query:
                self.logger.error("Unified input routed to standard query processing, but 'query' key is missing or empty.")
                return {"status": "error", "message": "Input for standard processing must contain a 'query'."}

            overrides = input_data.get("overrides") # Can be None
            return await self.process_query(query, query_data_overrides=overrides)


if __name__ == '__main__':
    print("--- Coordinator Basic Test ---")

    # Ensure JefeAgent can be imported for the __main__ test block
    # This might require specific path setup if running coordinator.py directly
    JefeAgent_class = None
    try:
        from src.agents.jefe_agent import JefeAgent as ImportedJefeAgent
        JefeAgent_class = ImportedJefeAgent
    except ImportError:
        print("WARNING: JefeAgent could not be imported for __main__ test. Real-time tests will be skipped or limited.")
        # Define a mock JefeAgent if the real one isn't available for testing basic coordinator flow
        class MockJefeAgent(BaseAgent):
            async def process_realtime_input(self, jefe_context: JefeContext) -> Dict[str, Any]:
                self.logger.info(f"MockJefeAgent processing real-time input: {jefe_context.summarize(50,30)}")
                return {"status": "success", "content": "Mock JefeAgent response to real-time input."}
        JefeAgent_class = MockJefeAgent


    def setup_jefe_integration(coordinator_instance: Coordinator):
        if not JefeAgent_class:
            print("JefeAgent class not available. Skipping Jefe registration in __main__.")
            return

        jefe_config = { # Simplified config for testing
            "model_for_tools": "gemini-1.5-flash-latest",
            "max_tokens_for_tools": 600,
            "temperature_for_tools": 0.25
        }
        try:
            jefe_agent_instance = JefeAgent_class( # type: ignore
                agent_name="JefeMainTest",
                api_manager=coordinator_instance.api_manager,
                config=jefe_config
            )
            coordinator_instance.register_jefe_agent(jefe_agent_instance) # type: ignore
            print("JefeAgent registered with Coordinator for __main__ test run.")
        except Exception as e:
            print(f"Error setting up JefeAgent for __main__ test: {e}", exc_info=True)


    try:
        coordinator = Coordinator()
        setup_jefe_integration(coordinator) # Attempt to register JefeAgent

        if not coordinator.agents and not coordinator.jefe_agent:
            print("No agents (general or Jefe) loaded. Coordinator tests will be limited.")
        else:
            print(f"General Agents: {list(coordinator.agents.keys())}")
            if coordinator.jefe_agent:
                 print(f"JefeAgent: {coordinator.jefe_agent.get_name()}")

            async def main_test():
                # Test 1: Standard query (handled by general agents or JefeAgent's process_query)
                query1 = "What is the capital of France? Explain in one sentence."
                print(f"\nProcessing standard query 1: '{query1}'")
                response1 = await coordinator.unified_process_input({"query": query1})
                print(f"Response 1: {response1}")
                print(f"--- Trace for query 1: '{query1[:50]}...' ---")
                for event in coordinator.context_manager.get_full_trace(): print(event)

                # Test 2: Another standard query
                query2 = "Write a Python function to calculate factorial."
                print(f"\nProcessing standard query 2: '{query2}'")
                response2 = await coordinator.unified_process_input({"query": query2})
                print(f"Response 2: {response2}")
                print(f"--- Trace for query 2: '{query2[:50]}...' ---")
                for event in coordinator.context_manager.get_full_trace(): print(event)

                # Test 3: Sequential plan query
                query_sequential = "summarize critique and list keywords for this document about AI ethics."
                print(f"\nProcessing sequential plan query: '{query_sequential}'")
                response_sequential = await coordinator.unified_process_input({"query": query_sequential})
                print(f"Response (sequential): {response_sequential}")
                print(f"--- Trace for query (sequential): '{query_sequential[:50]}...' ---")
                for event in coordinator.context_manager.get_full_trace(): print(event)

                # Test 4: Parallel plan query
                query_market = "concurrent market and competitor analysis for new EV startup"
                print(f"\nProcessing parallel plan query (market): '{query_market}'")
                response_market = await coordinator.unified_process_input({"query": query_market})
                print(f"Response (market): {response_market}")
                print(f"--- Trace for query (market): '{query_market[:50]}...' ---")
                for event in coordinator.context_manager.get_full_trace(): print(event)

                # Test 5: Real-time input example (should be routed to JefeAgent if registered)
                if coordinator.jefe_agent or "jefe_realtime_processor" in coordinator.realtime_processors:
                    realtime_input_example = {
                        'screen_content': "class Buggy {\n constructor() { console.log('error' \n} }", # Syntax error
                        'audio_transcript': "I have a syntax error in my JavaScript code, it's not working.",
                        'current_ide': "VSCode",
                        'programming_language': "JavaScript",
                        'error_messages': ["SyntaxError: Unexpected token '}'"],
                        'source': 'realtime' # Indicator for routing, or rely on key presence
                    }
                    print(f"\nProcessing real-time input via Jefe: {str(realtime_input_example)[:100]}...")
                    response_realtime = await coordinator.unified_process_input(realtime_input_example)
                    print(f"Response (real-time Jefe): {response_realtime}")
                    print(f"--- Trace for real-time Jefe input ---")
                    for event in coordinator.context_manager.get_full_trace(): print(event)
                else:
                    print("\nSkipping real-time input test as JefeAgent or its processor is not registered.")


            if os.name == 'nt':
                 asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(main_test())

    except Exception as e: print(f"Error in main test execution: {e}", exc_info=True)
    print("\n--- Coordinator Basic Test Finished ---")
