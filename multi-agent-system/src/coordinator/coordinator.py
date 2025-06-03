# src/coordinator/coordinator.py
"""
Main orchestrator for the multi-agent system.
Initializes agents, analyzes queries, routes tasks, and returns results.
"""

from typing import Dict, Any, List, Optional
import os 
import re 
import asyncio # Added for parallel execution

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
except ImportError:
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
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


class Coordinator:
    """
    Orchestrates task processing by analyzing queries, selecting appropriate
    agents, and dispatching tasks to them.
    """

    def __init__(self, agent_config_path: Optional[str] = None):
        self.logger = get_logger("Coordinator")
        self.api_manager = APIManager(config_path=agent_config_path)
        self.task_analyzer = TaskAnalyzer()
        self.routing_engine = RoutingEngine()
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_classes: Dict[str, type[BaseAgent]] = {
            "deepseek": DeepSeekAgent, "claude": ClaudeAgent,
            "cursor": CursorAgent, "windsurf": WindsurfAgent,
            "gemini": GeminiAgent,
        }
        self._instantiate_agents()
        if self.agents:
            self.logger.info("Coordinator initialized successfully with agents: " + ", ".join(self.agents.keys()))
        else:
            self.logger.warning("Coordinator initialized, but no agents were instantiated. Check configurations.")

    def register_agent(self, agent_instance: BaseAgent):
        agent_name = agent_instance.get_name()
        if agent_name in self.agents:
            self.logger.warning(f"Agent '{agent_name}' already registered. Overwriting with new instance.")
        self.agents[agent_name] = agent_instance
        self.logger.info(f"Agent '{agent_name}' registered successfully.")

    def _instantiate_agents(self):
        self.logger.info("Attempting to instantiate agents based on loaded configurations...")
        agent_configs = self.api_manager.service_configs
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
        self, mapping_config: Dict[str, Any], initial_query_input: str, 
        execution_context: Dict[str, Any], current_step_index: int, 
        full_plan_details: List[Dict[str, Any]],
        # For _execute_branch_sequentially, we need context *within* the branch
        # and the main execution_context for refs *outside* the branch.
        # Let's assume for now _resolve_input_value is only used for main plan or gets appropriate context.
        # For branch internal refs, _execute_branch_sequentially will manage its own small context.
        # For refs like {ref:step_before_parallel_block.field}, it needs the main context.
        # The `TaskAnalyzer` currently generates `ref:previous_step.content` which is relative.
        # And `original_query` which is absolute.
        # More complex refs like `ref:specific_step_id.field` are resolved against `execution_context`.
        # So, `execution_context` here should be the relevant one for the reference.
    ) -> Optional[Any]:
        if "value" in mapping_config: return mapping_config["value"]
        source_uri = mapping_config.get("source")
        if not source_uri:
            self.logger.warning(f"Input mapping {mapping_config} missing 'source' or 'value'. Returning None.")
            return None
        if source_uri == "original_query": return initial_query_input
        if source_uri.startswith("ref:"):
            ref_path = source_uri.split("ref:", 1)[1]
            target_step_output_id, field_path_str = "", "" # Initialize
            
            if ref_path == "previous_step.content":
                if current_step_index == 0:
                    self.logger.error("Cannot use 'ref:previous_step.content' for first step.")
                    return None
                # This previous_step_output_id is relative to the current list of steps being processed
                # (either main plan or a branch). `full_plan_details` should be that current list.
                target_step_output_id = full_plan_details[current_step_index - 1].get("output_id")
                if not target_step_output_id:
                    self.logger.error(f"Prev step (idx {current_step_index -1}) missing 'output_id'.")
                    return None
                field_path_str = "content"
            else:
                if '.' not in ref_path:
                    self.logger.error(f"Invalid ref path '{ref_path}'. Must be 'step_id.field_path'.")
                    return None
                target_step_output_id, field_path_str = ref_path.split('.', 1)

            target_resp_obj = execution_context.get(target_step_output_id)
            if target_resp_obj is None:
                self.logger.error(f"Ref'd step output '{target_step_output_id}' not in context.")
                return None
            if not isinstance(target_resp_obj, dict) or target_resp_obj.get("status") != "success":
                self.logger.error(f"Ref'd step '{target_step_output_id}' failed or invalid output: {target_resp_obj}")
                return None
            
            _NOT_FOUND = object()
            resolved_data = get_nested_value(target_resp_obj, field_path_str, default=_NOT_FOUND)
            if resolved_data is _NOT_FOUND:
                self.logger.error(f"Path '{field_path_str}' not found in output of step '{target_step_output_id}'.")
                return None
            return resolved_data
        
        if "template" in mapping_config:
            template_str = mapping_config["template"]
            def replacer(match: re.Match) -> str:
                if match.group(1) == "original_query": return initial_query_input
                ref_p = match.group(2)
                if '.' not in ref_p:
                    self.logger.warning(f"Template ref '{ref_p}' invalid format. Placeholder used.")
                    return "[INVALID_REF_FORMAT_IN_TEMPLATE]"
                step_id, field_p = ref_p.split('.', 1)
                resp_obj = execution_context.get(step_id)
                if resp_obj is None or not isinstance(resp_obj, dict) or resp_obj.get("status") != "success":
                    self.logger.warning(f"Template ref step '{step_id}' not found/failed. Placeholder used.")
                    return "[DATA_NOT_FOUND]"
                _NOT_FOUND_TPL = object()
                val = get_nested_value(resp_obj, field_p, default=_NOT_FOUND_TPL)
                if val is _NOT_FOUND_TPL:
                    self.logger.warning(f"Template ref path '{field_p}' in step '{step_id}' not found. Placeholder used.")
                    return "[DATA_NOT_FOUND]"
                return str(val)
            try:
                return re.sub(r"\{(original_query|ref:([^}]+))\}", replacer, template_str)
            except Exception as e:
                self.logger.error(f"Error processing template '{template_str}': {e}", exc_info=True)
                return None
        self.logger.warning(f"Unknown input mapping type in {mapping_config}. Returning None.")
        return None

    async def _execute_branch_sequentially(
        self, branch_step_templates: List[Dict], initial_branch_input: str, branch_id_for_logging: str,
        global_query_data_overrides: Optional[Dict[str, Any]],
        main_execution_context: Dict[str, Any] # Context from before the parallel block
    ) -> Optional[Dict[str, Any]]:
        self.logger.info(f"Executing branch '{branch_id_for_logging}' with {len(branch_step_templates)} steps.")
        branch_internal_context: Dict[str, Any] = {}
        last_branch_response: Optional[Dict[str, Any]] = None

        for i, step_def in enumerate(branch_step_templates):
            agent_name = step_def["agent_name"]
            if agent_name not in self.agents:
                self.logger.error(f"Agent '{agent_name}' in branch '{branch_id_for_logging}' not available. Branch fails.")
                return {"status": "error", "message": f"Agent '{agent_name}' for branch '{branch_id_for_logging}' not found."}
            current_agent = self.agents[agent_name]

            current_agent_inputs: Dict[str, Any] = {}
            input_mapping_config = step_def.get("input_mapping", {})
            
            # For _resolve_input_value, current_step_index is 'i' within the branch,
            # full_plan_details is 'branch_step_templates',
            # and execution_context is 'branch_internal_context' for 'previous_step' refs,
            # but 'main_execution_context' for other 'ref:step_id.field' that might point outside the branch.
            # This requires _resolve_input_value to handle two contexts or a merged view.
            # For now, 'ref:previous_step.content' uses branch_internal_context.
            # 'original_query' uses initial_branch_input.
            # Other 'ref:step_id.field' will use main_execution_context.
            # This is a simplification; more robust context management might be needed.

            for input_key, mapping_config in input_mapping_config.items():
                # Determine context for resolution
                effective_context = main_execution_context
                if mapping_config.get("source", "").startswith("ref:previous_step"):
                    effective_context = branch_internal_context

                resolved_value = self._resolve_input_value(
                    mapping_config, initial_branch_input, effective_context, i, branch_step_templates
                )
                if resolved_value is None and mapping_config.get("required", True):
                    msg = f"Critical input '{input_key}' unresolved in branch '{branch_id_for_logging}', step {i+1} ({agent_name})."
                    self.logger.error(msg)
                    return {"status": "error", "message": msg, "agent_name": agent_name}
                current_agent_inputs[input_key] = resolved_value
            
            agent_query_data = current_agent_inputs.copy()
            # Gemini & Overrides logic (same as in main sequential/single path)
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
            try:
                response = await current_agent.process_query(agent_query_data) # ASYNC CALL
            except Exception as e:
                msg = f"Exception in branch '{branch_id_for_logging}', step {i+1} ({agent_name}): {e}"
                self.logger.error(msg, exc_info=True)
                return {"status": "error", "message": msg, "agent_name": agent_name}
            
            if step_def.get("output_id"): branch_internal_context[step_def["output_id"]] = response
            last_branch_response = response
            if response.get("status") != "success":
                self.logger.warning(f"Agent {agent_name} in branch '{branch_id_for_logging}' returned error: {response.get('message')}")
                return response
            if response.get("content") is None and i < len(branch_step_templates) -1: # If not last step and no content to chain
                msg = f"Agent {agent_name} in branch '{branch_id_for_logging}' missing 'content' for chaining."
                self.logger.error(msg)
                return {"status": "error", "message": msg, "agent_name": agent_name}
            
            if i < len(branch_step_templates) -1 : # If there are more steps in this branch
                 current_input_content = str(response.get("content","")) # Update for next step in this branch

        self.logger.info(f"Branch '{branch_id_for_logging}' completed. Final response: {str(last_branch_response)[:100]}...")
        return last_branch_response


    async def process_query(self, query: str, query_data_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.logger.info(f"Coordinator received query: '{query[:100]}...'")
        if not self.agents:
            self.logger.error("No agents registered. Cannot process query.")
            return {"status": "error", "message": "No agents available."}

        analysis_result = self.task_analyzer.analyze_query(query, self.agents)
        self.logger.debug(f"Task analysis: {analysis_result}")
        
        # selected_agents from RoutingEngine are agent instances, ordered if from a plan
        # For parallel block, RoutingEngine should return a flat list of all agents involved across all branches.
        # The Coordinator then uses the execution_plan to structure calls.
        # Let's assume RoutingEngine returns agents listed in plan if plan exists.
        selected_agent_instances_from_router = self.routing_engine.select_agents(analysis_result, self.agents)
        
        if not selected_agent_instances_from_router: # No agents selected by router (plan invalid or no suggestions/fallback)
            self.logger.warning("Router selected no agents.")
            return {"status": "error", "message": "No suitable agent found."}

        execution_plan_details = analysis_result.get("execution_plan")
        initial_query_input = str(analysis_result.get("processed_query_for_agent", query))
        execution_context: Dict[str, Any] = {}
        final_response: Optional[Dict[str, Any]] = None

        # If plan is a parallel block
        if execution_plan_details and len(execution_plan_details) == 1 and execution_plan_details[0].get("type") == "parallel_block":
            parallel_block_def = execution_plan_details[0]
            self.logger.info(f"Executing parallel block: {parallel_block_def.get('task_description')}")
            
            branch_templates = parallel_block_def.get("branches", [])
            branch_coroutines = []

            # The `selected_agent_instances_from_router` needs to contain all unique agents
            # mentioned in all branches of the parallel_block_def for _execute_branch_sequentially to find them.
            # RoutingEngine should ideally provide this flat list if it validated the parallel plan.

            for idx, branch_step_templates in enumerate(branch_templates):
                if not branch_step_templates: continue # Skip empty branch

                # For parallel branches, initial input for each branch's first step typically comes from the overall query
                # or a step preceding the parallel block. TaskAnalyzer sets first step of branch to "original_query".
                # The _resolve_input_value in _execute_branch_sequentially will handle this.
                # We pass main `initial_query_input` and `execution_context` (from before this parallel block).
                branch_coroutines.append(
                    self._execute_branch_sequentially(
                        branch_step_templates, 
                        initial_query_input, # Main initial input for "original_query" refs
                        f"branch_{idx}", 
                        query_data_overrides,
                        execution_context.copy() # Pass context from *before* this parallel block
                    )
                )
            
            if not branch_coroutines:
                self.logger.warning("Parallel block defined but has no branches to execute.")
                return {"status": "error", "message": "Parallel block has no executable branches."}

            branch_execution_results = await asyncio.gather(*branch_coroutines, return_exceptions=True)
            
            aggregated_results_data: Dict[str, Any] = {}
            all_branches_succeeded = True
            for idx, result in enumerate(branch_execution_results):
                branch_key = f"branch_{idx}_result" # Store under a key indicating the branch
                if isinstance(result, Exception):
                    aggregated_results_data[branch_key] = {"status": "error", "message": f"Branch execution raised: {str(result)}", "details": str(result)}
                    all_branches_succeeded = False
                elif result is None or result.get("status") != "success":
                    aggregated_results_data[branch_key] = result or {"status": "error", "message": "Branch failed or no data"}
                    all_branches_succeeded = False
                else:
                    aggregated_results_data[branch_key] = result
            
            pb_output_id = parallel_block_def.get("output_id", "parallel_block_output")
            current_step_result = {
                "status": "success" if all_branches_succeeded else "partial_success", 
                "type": "parallel_block_result",
                "output_id": pb_output_id, # The parallel block itself has an output_id
                "aggregated_results": aggregated_results_data
            }
            execution_context[pb_output_id] = current_step_result 
            final_response = current_step_result
            # If this parallel block is the only "step" in execution_plan_details, this is the final response.
            # If there are more steps *after* this parallel block in execution_plan_details (not supported by current TaskAnalyzer output):
            # The next step would use `ref:pb_output_id.aggregated_results.branch_X_result.content` etc.

        # Sequential plan (list of steps) or single agent from suggestion
        elif execution_plan_details: # It's a sequential plan (list of steps)
            self.logger.info(f"Starting rich sequential execution for {len(selected_agent_instances_from_router)} agents based on plan.")
            # `selected_agent_instances_from_router` should match the agents in `execution_plan_details` in order.
            for i, step_def in enumerate(execution_plan_details):
                # ... (Existing rich sequential logic from previous step, now needs await) ...
                # This part needs to be merged with the _execute_branch_sequentially logic almost,
                # but using main execution_context and selected_agent_instances_from_router.
                # For now, let's adapt the _execute_branch_sequentially for a single "branch" (the main plan)
                # This needs a slight refactor. Let's assume _execute_branch_sequentially can run the main plan.
                # This is becoming complex. The original plan had a main loop that could handle parallel blocks
                # as one type of step. Let's revert to that structure.

                # --- Re-inserting main plan loop logic here, adapted for async and parallel block handling ---
                # This section is for sequential plans or single steps that are NOT parallel blocks.
                # If a parallel block was the only item in execution_plan_details, it's handled above.
                # If execution_plan_details has multiple items, and one is a parallel_block, this loop structure is needed.
                # This means the parallel block handler should be *inside* a main loop for execution_plan_details.

                # For now, assume execution_plan_details is either a single parallel_block OR a list of sequential steps.
                # The code above handles the parallel_block if it's the content of execution_plan_details[0].
                # If it was a sequential plan, we need a sequential loop here.
                # The current `elif len(selected_agents) > 1 and analysis_result.get("execution_plan")` implies
                # `selected_agents` is already the list for the plan.

                # This code path is for a *sequential plan* (not a parallel block as the primary step type)
                # This logic is very similar to _execute_branch_sequentially but uses main `execution_context`
                if i >= len(selected_agent_instances_from_router):
                     return {"status": "error", "message": "Plan steps and selected agents mismatch."}
                current_agent = selected_agent_instances_from_router[i]
                # ... (Input resolution using _resolve_input_value, initial_query_input, execution_context, i, execution_plan_details)
                # ... (Prepare agent_query_data with overrides)
                # ... (Call await current_agent.process_query)
                # ... (Store in execution_context, update final_response, handle errors)
                # This is essentially what _execute_branch_sequentially does.
                # So, a sequential plan IS a single branch.
                # We can call _execute_branch_sequentially for it.
                
                # This means the main `process_query` needs to differentiate:
                # 1. Is it a single parallel_block step? -> Use parallel logic.
                # 2. Is it a list of sequential steps? -> Use sequential logic (can use _execute_branch_sequentially).
                # 3. Is it a single agent from suggestion (no plan)? -> Use single agent logic.

                # For this iteration, if it's a plan, and not a parallel block as the first step,
                # we run it sequentially.
                final_response = await self._execute_branch_sequentially(
                    execution_plan_details, initial_query_input, "main_sequence",
                    query_data_overrides, execution_context # execution_context starts empty here
                )
                # The `execution_context` would be populated by `_execute_branch_sequentially` if it was passed by ref.
                # But it's not, so this won't work for multi-step main sequences if `_execute_branch_sequentially`
                # relies on an external context for non-branch-internal refs.
                # This indicates `_execute_branch_sequentially` needs to be more general or
                # the main loop needs to be here.

                # Let's put the main sequential loop here for clarity.
                # This effectively means `_execute_branch_sequentially` is only for *branches* of a parallel block.
                # And the main `execution_plan` (if sequential) is handled by a loop here.

                # --- START REVISED MAIN SEQUENTIAL LOOP (if not parallel block) ---
                self.logger.info(f"Starting sequential execution for plan: {[s.get('agent_name') for s in execution_plan_details]}")
                temp_execution_context = {} # Context for this sequence
                current_input_for_sequence = initial_query_input # Start with initial query for first step if it maps to original_query

                for step_idx, step_detail in enumerate(execution_plan_details):
                    seq_agent_name = step_detail["agent_name"]
                    if seq_agent_name not in self.agents:
                        return {"status": "error", "message": f"Agent '{seq_agent_name}' in sequential plan not found."}
                    seq_current_agent = self.agents[seq_agent_name]
                    
                    seq_agent_inputs = {}
                    for input_key, map_cfg in step_detail.get("input_mapping", {}).items():
                        # For sequential plan, context is temp_execution_context.
                        # `initial_query_input` is the global one.
                        # `step_idx` is current index, `execution_plan_details` is the plan.
                        resolved_val = self._resolve_input_value(map_cfg, initial_query_input, temp_execution_context, step_idx, execution_plan_details)
                        if resolved_val is None and map_cfg.get("required", True):
                            return {"status": "error", "message": f"Failed to resolve input '{input_key}' for {seq_agent_name}."}
                        seq_agent_inputs[input_key] = resolved_val
                    
                    seq_agent_query_data = seq_agent_inputs.copy()
                    # Gemini & Overrides (similar to _execute_branch_sequentially)
                    gemini_cls = self.agent_classes.get("gemini")
                    is_gem = gemini_cls and isinstance(seq_current_agent, gemini_cls)
                    if is_gem:
                        if "prompt" in seq_agent_query_data and seq_agent_query_data["prompt"] is not None:
                            seq_agent_query_data["prompt_parts"] = [str(seq_agent_query_data.pop("prompt"))]
                        elif "prompt_parts" not in seq_agent_query_data: self.logger.warning("Gemini agent in seq plan missing prompt.")
                    elif "prompt_parts" in seq_agent_query_data: self.logger.warning("Non-Gemini agent in seq plan got prompt_parts.")

                    eff_cfg = {}
                    if query_data_overrides: eff_cfg.update(query_data_overrides.copy())
                    if step_detail.get("agent_config_overrides"): eff_cfg.update(step_detail["agent_config_overrides"])
                    for key in seq_agent_inputs.keys(): eff_cfg.pop(key, None)
                    if "system_prompt" in eff_cfg: seq_agent_query_data["system_prompt"] = eff_cfg.pop("system_prompt")
                    elif step_idx == 0 and analysis_result.get("system_prompt") and "system_prompt" not in seq_agent_query_data:
                        seq_agent_query_data["system_prompt"] = analysis_result.get("system_prompt")
                    seq_agent_query_data.update(eff_cfg)

                    self.logger.debug(f"Sequential plan step {step_idx+1} ({seq_agent_name}) data: {str(seq_agent_query_data)[:200]}...")
                    try:
                        step_resp = await seq_current_agent.process_query(seq_agent_query_data)
                    except Exception as e:
                        return {"status": "error", "message": f"Exception in seq plan step {step_idx+1} ({seq_agent_name}): {e}", "agent_name": seq_agent_name}
                    
                    if step_detail.get("output_id"): temp_execution_context[step_detail["output_id"]] = step_resp
                    final_response = step_resp # Last response is the final one for the sequence
                    if step_resp.get("status") != "success": return step_resp
                    if step_resp.get("content") is None and step_idx < len(execution_plan_details) -1:
                         return {"status": "error", "message": f"Agent {seq_agent_name} in seq plan missing 'content'.", "agent_name": seq_agent_name}
                    # current_input_for_sequence = str(step_resp.get("content","")) # This was implicit, now explicit via input_mapping
                # End of sequential plan loop
                # --- END REVISED MAIN SEQUENTIAL LOOP ---
        
        elif len(selected_agent_instances_from_router) == 1: # Single agent suggested, no plan
            primary_agent = selected_agent_instances_from_router[0]
            # ... (existing single agent logic, now needs await) ...
            agent_query_data: Dict[str, Any] = {}
            processed_prompt = analysis_result.get("processed_query_for_agent", query)
            gemini_agent_class = self.agent_classes.get("gemini")
            is_gemini_instance = gemini_agent_class and isinstance(primary_agent, gemini_agent_class)
            if is_gemini_instance: agent_query_data["prompt_parts"] = [processed_prompt]
            else: agent_query_data["prompt"] = processed_prompt
            if analysis_result.get("system_prompt"): agent_query_data["system_prompt"] = analysis_result.get("system_prompt")
            if query_data_overrides: # Apply overrides
                temp_overrides = query_data_overrides.copy()
                if is_gemini_instance:
                    if "prompt_parts" in temp_overrides: agent_query_data["prompt_parts"] = temp_overrides.pop("prompt_parts")
                    if "prompt" in temp_overrides and "prompt_parts" in agent_query_data: temp_overrides.pop("prompt", None)
                else: 
                    if "prompt" in temp_overrides: agent_query_data["prompt"] = temp_overrides.pop("prompt")
                    if "prompt_parts" in temp_overrides: temp_overrides.pop("prompt_parts", None)
                agent_query_data.update(temp_overrides)
            self.logger.debug(f"Single agent query data for {primary_agent.get_name()}: {agent_query_data}")
            try:
                final_response = await primary_agent.process_query(agent_query_data) # ASYNC CALL
            except Exception as e:
                self.logger.error(f"Error with single agent {primary_agent.get_name()}: {e}", exc_info=True)
                return {"status": "error", "message": f"Failed query with {primary_agent.get_name()}: {str(e)}"}

        else: # Fallback to simple sequential (legacy, if router returns multiple agents but TaskAnalyzer had no plan)
            self.logger.warning(f"Executing legacy simple sequential for {len(selected_agent_instances_from_router)} agents.")
            current_input_content = initial_query_input
            legacy_final_response: Dict[str, Any] = {}
            for i, agent_instance in enumerate(selected_agent_instances_from_router):
                self.logger.info(f"Legacy sequential step {i+1}: {agent_instance.get_name()}")
                agent_query_data = {"prompt": current_input_content}
                if query_data_overrides:
                    temp_overrides = query_data_overrides.copy()
                    temp_overrides.pop("prompt", None); temp_overrides.pop("prompt_parts", None)
                    agent_query_data.update(temp_overrides)
                try:
                    response = await agent_instance.process_query(agent_query_data) # ASYNC CALL
                    legacy_final_response = response
                    if response.get("status") != "success" or response.get("content") is None:
                        return response
                    current_input_content = str(response.get("content"))
                except Exception as e:
                    return {"status": "error", "message": f"Error in legacy seq with {agent_instance.get_name()}: {str(e)}"}
            final_response = legacy_final_response
        
        return final_response if final_response is not None else {"status": "error", "message": "Processing resulted in no final response."}


if __name__ == '__main__':
    print("--- Coordinator Basic Test ---")
    # ... (rest of __main__ block, needs async await for coordinator.process_query)
    # For instance: asyncio.run(coordinator.process_query(...))
    # This __main__ block needs significant update to run async code.
    # For brevity, I will omit fully converting it here, as it's for basic demo.
    # The unit tests are the primary way to test async functionality.
    # ... (existing __main__ setup code) ...
    try:
        coordinator = Coordinator() # agent_config_path=dummy_config_file omitted for less verbose diff
        if not coordinator.agents: print("No agents loaded.")
        else:
            print(f"Agents: {list(coordinator.agents.keys())}")
            # Example of running an async method from sync context (for __main__ only)
            # In a real async app, you'd await it directly.
            async def main_test():
                query1 = "What is the capital of France? Explain in one sentence."
                print(f"\nProcessing query 1: '{query1}'")
                response1 = await coordinator.process_query(query1)
                print(f"Response 1: {response1}")

                query2 = "Write a Python function to calculate factorial."
                print(f"\nProcessing query 2: '{query2}'")
                response2 = await coordinator.process_query(query2)
                print(f"Response 2: {response2}")

                # Example for a plan that might be parallel (if TaskAnalyzer is configured for it)
                # query_market = "concurrent market and competitor analysis for new EV startup"
                # print(f"\nProcessing query (market): '{query_market}'")
                # response_market = await coordinator.process_query(query_market)
                # print(f"Response (market): {response_market}")

            if os.name == 'nt': # Windows patch for asyncio if needed for basic demo run
                 asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(main_test())

    except Exception as e: print(f"Error in main: {e}")
    print("\n--- Coordinator Basic Test Finished ---")
