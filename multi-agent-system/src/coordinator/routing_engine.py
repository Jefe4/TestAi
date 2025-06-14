# src/coordinator/routing_engine.py
"""
Determines which agent(s) should handle a task based on analysis results.

The RoutingEngine is responsible for interpreting the 'execution_plan' or
'suggested_agents' fields from the TaskAnalyzer's output and selecting
the appropriate, available agent instances.
"""

from typing import Dict, Any, List, Optional # Standard typing modules

try:
    from ..agents.base_agent import BaseAgent
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for scenarios where the module might be run directly for testing
    # or if the PYTHONPATH is not set up correctly during development.
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.logger import get_logger # type: ignore

class RoutingEngine:
    """
    Selects appropriate agent(s) to handle a query based on the analysis
    provided by the TaskAnalyzer and the list of available agents.
    It first checks for a valid 'execution_plan' (either parallel or sequential).
    If no valid plan is found, it falls back to 'suggested_agents'.
    As a final fallback, it can select the first available agent if configured to do so.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the RoutingEngine.

        Args:
            config: Optional configuration dictionary for the engine.
                    Example: `{"fallback_to_first_available": True}`.
                    If `fallback_to_first_available` is True (default), the engine
                    will select the first available agent if no other selection logic yields results.
        """
        self.config = config if config is not None else {}
        self.logger = get_logger("RoutingEngine")
        self.logger.info(f"RoutingEngine initialized with config: {self.config}")

    def select_agents(self, analysis_result: Dict[str, Any], available_agents: Dict[str, BaseAgent]) -> List[BaseAgent]:
        """
        Selects a list of agent instances based on the analysis result from TaskAnalyzer.

        The selection process prioritizes:
        1. A valid "execution_plan" (parallel or sequential) from `analysis_result`.
           - For parallel plans, it extracts all unique agent names from all branches.
           - For sequential plans, it extracts agent names in the specified order.
           If any agent in the plan is not available, the plan is considered invalid,
           and an empty list is returned (no fallback to suggestions from an invalid plan).
        2. If no valid execution plan, it uses "suggested_agents" from `analysis_result`.
        3. If no agents are selected via plan or suggestions, it applies fallback logic:
           - If `self.config.get("fallback_to_first_available", True)` is set,
             it selects the first agent from `available_agents`.

        Args:
            analysis_result: The output dictionary from `TaskAnalyzer.analyze_query`.
                             This may contain "execution_plan" (structured plan) or
                             "suggested_agents" (list of agent names).
            available_agents: A dictionary of currently available agent instances,
                              keyed by their names (e.g., "ClaudeAgent", "DeepSeekAgent").

        Returns:
            A list of `BaseAgent` instances. If a plan was used, the order of agents
            (for sequential plans) or the set of unique agents (for parallel plans)
            is returned. If no agents are selected, an empty list is returned.
        """
        query_type = analysis_result.get('query_type', 'N/A') # For logging purposes
        execution_plan = analysis_result.get("execution_plan") # Can be None, empty, or a structured plan

        self.logger.info(
            f"Selecting agents for query type '{query_type}'. Available agents: {list(available_agents.keys())}."
        )

        if execution_plan and isinstance(execution_plan, list) and len(execution_plan) > 0:
            self.logger.info(f"TaskAnalyzer provided an execution plan: {str(execution_plan)[:200]}...") # Log truncated plan

            # --- Handle Parallel Block Plan ---
            # A parallel plan is expected to be a list containing a single dictionary of type "parallel_block".
            if isinstance(execution_plan[0], dict) and execution_plan[0].get("type") == "parallel_block":
                self.logger.info("Processing as a 'parallel_block' execution plan.")
                unique_agent_names_in_plan: Set[str] = set()
                parallel_block_def = execution_plan[0]
                branches = parallel_block_def.get("branches", [])

                if not branches:
                    self.logger.warning("Parallel_block found, but no 'branches' are defined. Plan is invalid.")
                    return [] # Invalid plan, return empty

                # Collect all unique agent names from all steps in all branches
                for branch_idx, branch_steps in enumerate(branches):
                    if not isinstance(branch_steps, list):
                        self.logger.warning(f"Invalid branch format in parallel_block (branch {branch_idx}): {branch_steps}. Expected a list of steps. Plan is invalid.")
                        return [] # Invalid plan structure
                    for step_idx, step_definition in enumerate(branch_steps):
                        if not isinstance(step_definition, dict) or "agent_name" not in step_definition:
                            self.logger.warning(
                                f"Invalid step_definition in parallel_block branch {branch_idx}, step {step_idx}: {step_definition}. "
                                "Must be a dict with 'agent_name'. Plan is invalid."
                            )
                            return [] # Invalid step structure
                        unique_agent_names_in_plan.add(step_definition["agent_name"])

                # Retrieve agent instances for all unique names
                planned_agent_instances: List[BaseAgent] = []
                all_agents_found_for_parallel = True
                for agent_name in list(unique_agent_names_in_plan): # Convert set to list for iteration
                    agent_instance = available_agents.get(agent_name)
                    if agent_instance:
                        planned_agent_instances.append(agent_instance)
                    else:
                        self.logger.warning(
                            f"Agent '{agent_name}' from parallel_block execution_plan not found in available_agents. "
                            "Parallel plan is invalid."
                        )
                        all_agents_found_for_parallel = False
                        break # Stop if any agent is missing

                if all_agents_found_for_parallel:
                    self.logger.info(
                        f"Parallel_block plan is valid. Selected {len(planned_agent_instances)} unique agent(s): "
                        f"{[agent.get_name() for agent in planned_agent_instances]}"
                    )
                    # For parallel execution, the Coordinator will use the plan structure to dispatch tasks.
                    # The list returned here contains all agents needed for the parallel block.
                    return planned_agent_instances
                else:
                    # If any agent in a parallel plan is not found, the plan is considered invalid.
                    self.logger.warning("Parallel_block plan was invalid (one or more agents not found). Returning no agents.")
                    return []

            # --- Handle Sequential Plan ---
            # A sequential plan is a list of step definition dictionaries.
            # Each step dictionary must have an "agent_name".
            elif isinstance(execution_plan[0], dict) and "agent_name" in execution_plan[0]:
                self.logger.info("Processing as a sequential execution plan (list of step definitions).")
                planned_agent_instances: List[BaseAgent] = []
                is_sequential_plan_valid = True
                for step_idx, step_definition in enumerate(execution_plan):
                    if not isinstance(step_definition, dict) or "agent_name" not in step_definition:
                        self.logger.warning(
                            f"Invalid step_definition in sequential plan (step {step_idx}): {step_definition}. "
                            "Must be a dict with 'agent_name'. Plan is invalid."
                        )
                        is_sequential_plan_valid = False
                        break
                    agent_name = step_definition["agent_name"]
                    agent_instance = available_agents.get(agent_name)
                    if agent_instance:
                        planned_agent_instances.append(agent_instance)
                    else:
                        self.logger.warning(
                            f"Agent '{agent_name}' from sequential execution_plan (step {step_idx}) not found in available_agents. "
                            "Sequential plan is invalid."
                        )
                        is_sequential_plan_valid = False
                        break

                if is_sequential_plan_valid:
                    self.logger.info(
                        f"Sequential execution plan is valid. Selected {len(planned_agent_instances)} agent(s) in planned order: "
                        f"{[agent.get_name() for agent in planned_agent_instances]}"
                    )
                    return planned_agent_instances # Return agents in the order specified by the plan
                else:
                    # If any agent in a sequential plan is not found, the plan is invalid.
                    self.logger.warning("Sequential execution plan was invalid (one or more agents not found or invalid step structure). Returning no agents.")
                    return []
            else:
                # This handles cases where execution_plan is not empty but doesn't match known structures
                # (e.g., old format like a list of strings, or malformed new format).
                self.logger.warning(
                    f"Execution plan format is not recognized or is invalid: {str(execution_plan)[:200]}. "
                    "Expected a parallel_block or a list of step_definition dicts. Returning no agents."
                )
                return [] # Unrecognized plan format is treated as invalid

        # --- Fallback to Suggestion Logic ---
        # This section is reached if:
        #   - `execution_plan` was None, empty.
        #   - `execution_plan` was deemed invalid by the logic above (and returned [] which then leads here if not caught).
        #     (Corrected: Invalid plans should return [] and not fall through here).
        # The prompt implies invalid plans should return empty, not fall through.
        # The current code structure means if plan processing above returns [], this part is skipped.
        # This is correct. This part runs if `execution_plan` was initially falsey.

        self.logger.info("No valid execution plan provided or plan was empty. Attempting to use 'suggested_agents'.")
        suggested_agent_names = analysis_result.get("suggested_agents", []) # Get list of suggested agent names
        self.logger.info(f"TaskAnalyzer suggested_agents: {suggested_agent_names}.")
        
        selected_agent_instances: List[BaseAgent] = []

        if not available_agents: # Should ideally not happen if Coordinator has agents
            self.logger.warning("No agents available in the system for routing. Returning empty list.")
            return []

        if suggested_agent_names: # If TaskAnalyzer provided suggestions
            for agent_name in suggested_agent_names:
                if agent_name in available_agents:
                    selected_agent_instances.append(available_agents[agent_name])
                    self.logger.debug(f"Agent '{agent_name}' added to selection from TaskAnalyzer's suggestions.")
                else:
                    self.logger.warning(
                        f"TaskAnalyzer suggested agent '{agent_name}', but it was not found in available agents. Skipping."
                    )
        else:
            self.logger.info("TaskAnalyzer provided no specific agent suggestions (and no valid plan was processed).")
        
        # --- Fallback to First Available Agent (if no plan and no valid suggestions) ---
        if not selected_agent_instances: # If still no agents selected after checking suggestions
            self.logger.info(
                "No agents selected based on execution plan or TaskAnalyzer's direct suggestions. "
                "Considering fallback options based on configuration."
            )
            
            # Check configuration if fallback to the first available agent is enabled
            if self.config.get("fallback_to_first_available", True): # Default to True if not specified
                if available_agents: 
                    # Get the name and instance of the first agent in the available_agents dict
                    # Note: Dictionary order is guaranteed from Python 3.7+
                    first_available_agent_name = list(available_agents.keys())[0] 
                    first_agent_instance = available_agents[first_available_agent_name]
                    selected_agent_instances.append(first_agent_instance)
                    self.logger.info(
                        f"Fallback to first available agent is enabled: Selected '{first_agent_instance.get_name()}'."
                    )
                else:
                    # This case should ideally be caught earlier (no available_agents at all)
                    self.logger.warning("Fallback to first available agent configured, but no agents are available in the system.")
            else:
                self.logger.info("Fallback to first available agent is disabled by configuration. No agents will be selected.")
        
        # Final check: if still no agents, log an error and return empty
        if not selected_agent_instances:
            self.logger.error(
                "RoutingEngine could not select any agent. No valid plan, no valid suggestions, and fallback (if enabled) yielded no agent."
            )
            return [] # Return empty list if no agent could be selected by any means

        # Log the final list of selected agents (from suggestions or fallback)
        selected_names = [agent.get_name() for agent in selected_agent_instances]
        self.logger.info(f"RoutingEngine finalized selection (suggestion/fallback-based). Selected {len(selected_names)} agent(s): {selected_names}")
        
        return selected_agent_instances


if __name__ == '__main__':
    # This block is for basic demonstration and testing of the RoutingEngine.
    # It uses a MockAgent to simulate agent capabilities and availability.

    class MockAgent(BaseAgent):
        """A mock agent class for testing purposes."""
        def __init__(self, name, capabilities_list: Optional[List[str]] = None):
            super().__init__(name, {}) # Call parent constructor
            self._capabilities = {"capabilities": capabilities_list if capabilities_list is not None else []}
            self.logger = get_logger(f"MockAgent.{name}") # Each mock agent gets its own logger instance

        async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]: # Made async
            """Simulates processing a query."""
            self.logger.info(f"MockAgent '{self.agent_name}' received query: {query_data}")
            return {"status": "success", "content": f"Mock response from {self.agent_name}"}

        def get_capabilities(self) -> Dict[str, Any]: 
            """Returns the mock capabilities of this agent."""
            return self._capabilities

    # Create mock agent instances
    agent_a = MockAgent("AgentA", ["general_analysis", "text_generation"])
    agent_b = MockAgent("AgentB", ["code_generation", "text_generation"])
    agent_c = MockAgent("AgentC", ["summarization"])
    
    mock_available_agents: Dict[str, BaseAgent] = {"AgentA": agent_a, "AgentB": agent_b, "AgentC": agent_c}

    # Initialize RoutingEngines with different fallback configurations
    engine_with_fallback = RoutingEngine(config={"fallback_to_first_available": True})
    engine_no_fallback = RoutingEngine(config={"fallback_to_first_available": False})

    print("--- Testing RoutingEngine with Various Execution Plan Structures ---")

    # Test Case 1: Valid Sequential Plan (Old format - list of strings, will be rejected by new logic)
    # This test is expected to FAIL plan parsing and return [] if strict, or fall to suggestions.
    # Current logic will reject this plan format.
    analysis_seq_plan_valid_old_format = {"execution_plan": ["AgentC", "AgentA"]}
    selected_seq_old = engine_no_fallback.select_agents(analysis_seq_plan_valid_old_format, mock_available_agents)
    print(f"Old Format Sequential Plan Selected: {[a.get_name() for a in selected_seq_old]}")
    # Expected: [] (because plan format is invalid)
    assert len(selected_seq_old) == 0, "Old format plan should be rejected"


    # Test Case 2: Valid Sequential Plan (New format - list of dicts)
    analysis_seq_plan_valid_new_format = {
        "execution_plan": [
            {"agent_name": "AgentC", "task_description": "Summarize"},
            {"agent_name": "AgentA", "task_description": "Analyze summary"}
        ]
    }
    selected_seq_new = engine_no_fallback.select_agents(analysis_seq_plan_valid_new_format, mock_available_agents)
    print(f"New Format Sequential Plan Selected: {[a.get_name() for a in selected_seq_new]}")
    assert [a.get_name() for a in selected_seq_new] == ["AgentC", "AgentA"], "New format sequential plan not processed correctly"

    # Test Case 3: Invalid Sequential Plan (AgentX not available)
    analysis_seq_plan_invalid_agent = {
         "execution_plan": [
            {"agent_name": "AgentA", "task_description": "Task 1"},
            {"agent_name": "AgentX", "task_description": "Task 2"}
        ]
    }
    selected_seq_invalid_agent = engine_no_fallback.select_agents(analysis_seq_plan_invalid_agent, mock_available_agents)
    print(f"Sequential Plan with Invalid Agent Selected: {[a.get_name() for a in selected_seq_invalid_agent]}")
    assert len(selected_seq_invalid_agent) == 0, "Plan with invalid agent should return empty"

    # Test Case 4: Valid Parallel Plan
    analysis_parallel_plan_valid = {
        "execution_plan": [{
            "type": "parallel_block",
            "branches": [
                [{"agent_name": "AgentA", "task_description": "Branch A Task 1"}],
                [{"agent_name": "AgentB", "task_description": "Branch B Task 1"}, {"agent_name": "AgentC", "task_description": "Branch B Task 2"}]
            ]
        }]
    }
    selected_parallel_valid = engine_no_fallback.select_agents(analysis_parallel_plan_valid, mock_available_agents)
    selected_parallel_names = sorted([a.get_name() for a in selected_parallel_valid]) # Sort for consistent comparison
    print(f"Valid Parallel Plan Selected Agents (sorted): {selected_parallel_names}")
    assert selected_parallel_names == ["AgentA", "AgentB", "AgentC"], "Parallel plan did not select all unique agents"

    # Test Case 5: Invalid Parallel Plan (AgentY not available)
    analysis_parallel_invalid_agent = {
        "execution_plan": [{
            "type": "parallel_block",
            "branches": [
                [{"agent_name": "AgentA", "task_description": "Task"}],
                [{"agent_name": "AgentY", "task_description": "Task"}]
            ]
        }]
    }
    selected_parallel_invalid = engine_no_fallback.select_agents(analysis_parallel_invalid_agent, mock_available_agents)
    print(f"Parallel Plan with Invalid Agent Selected: {[a.get_name() for a in selected_parallel_invalid]}")
    assert len(selected_parallel_invalid) == 0, "Parallel plan with invalid agent should return empty"


    print("\n--- Testing Suggestion Logic (No Execution Plan or Empty Plan) ---")
    # Test Case 6: Empty Plan - should fall through to suggestion logic
    analysis_empty_plan_with_sugg = {"execution_plan": [], "suggested_agents": ["AgentB"]}
    selected_empty_plan_sugg = engine_no_fallback.select_agents(analysis_empty_plan_with_sugg, mock_available_agents)
    print(f"Empty Plan (uses suggestions): {[a.get_name() for a in selected_empty_plan_sugg]}")
    assert [a.get_name() for a in selected_empty_plan_sugg] == ["AgentB"], "Empty plan did not fall back to suggestions correctly"

    # Test Case 7: No Plan, Valid Suggestions
    analysis_sugg_only_valid = {"suggested_agents": ["AgentB", "AgentC"]}
    selected_sugg_valid = engine_no_fallback.select_agents(analysis_sugg_only_valid, mock_available_agents)
    print(f"Suggestion Only (Valid) Selected: {[a.get_name() for a in selected_sugg_valid]}")
    assert [a.get_name() for a in selected_sugg_valid] == ["AgentB", "AgentC"], "Valid suggestions not processed correctly"
    
    # Test Case 8: No Plan, Invalid Suggestions, Fallback Enabled
    analysis_sugg_invalid_fb_enabled = {"suggested_agents": ["AgentY", "AgentZ"]}
    selected_sugg_invalid_fb = engine_with_fallback.select_agents(analysis_sugg_invalid_fb_enabled, mock_available_agents)
    print(f"Suggestion Invalid (Fallback Enabled): {[a.get_name() for a in selected_sugg_invalid_fb]}")
    assert [a.get_name() for a in selected_sugg_invalid_fb] == ["AgentA"], "Fallback to first available not working with invalid suggestions"

    # Test Case 9: No Plan, No Suggestions, Fallback Enabled
    analysis_no_sugg_fb_enabled = {"suggested_agents": []} # Or key missing
    selected_no_sugg_fb = engine_with_fallback.select_agents(analysis_no_sugg_fb_enabled, mock_available_agents)
    print(f"No Suggestions (Fallback Enabled): {[a.get_name() for a in selected_no_sugg_fb]}")
    assert [a.get_name() for a in selected_no_sugg_fb] == ["AgentA"], "Fallback to first available not working with no suggestions"

    # Test Case 10: No Plan, No Suggestions, Fallback Disabled
    analysis_no_sugg_no_fb = {"suggested_agents": []}
    selected_no_sugg_no_fb_disabled = engine_no_fallback.select_agents(analysis_no_sugg_no_fb, mock_available_agents)
    print(f"No Suggestions (Fallback Disabled): {[a.get_name() for a in selected_no_sugg_no_fb_disabled]}")
    assert len(selected_no_sugg_no_fb_disabled) == 0, "Should return empty when no suggestions and fallback disabled"
    
    print("\n--- RoutingEngine testing completed. ---")
