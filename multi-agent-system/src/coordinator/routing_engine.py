# src/coordinator/routing_engine.py
"""
Determines which agent(s) should handle a task based on analysis.
"""

from typing import Dict, Any, List, Optional

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
    Prioritizes an 'execution_plan' if provided, otherwise uses 'suggested_agents'.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the RoutingEngine.

        Args:
            config: Optional configuration dictionary for the engine.
                    Example: {"fallback_to_first_available": True/False}
        """
        self.config = config if config is not None else {}
        self.logger = get_logger("RoutingEngine")
        self.logger.info(f"RoutingEngine initialized with config: {self.config}")

    def select_agents(self, analysis_result: Dict[str, Any], available_agents: Dict[str, BaseAgent]) -> List[BaseAgent]:
        """
        Selects a list of agent instances based on the analysis result.
        Prioritizes "execution_plan" if present and valid. Otherwise, uses "suggested_agents"
        and then fallbacks.

        Args:
            analysis_result: The output from TaskAnalyzer.analyze_query.
                             May contain "execution_plan" (list of agent names for sequence)
                             or "suggested_agents" (list of agent names for single/parallel).
            available_agents: A dictionary of currently available agent instances,
                              keyed by their names.

        Returns:
            A list of BaseAgent instances deemed suitable for the query, in order if from a plan.
        """
        query_type = analysis_result.get('query_type', 'N/A')
        execution_plan = analysis_result.get("execution_plan") # This is a list of agent names

        self.logger.info(
            f"Selecting agents for query type '{query_type}'. Available agents: {list(available_agents.keys())}."
        )
        if execution_plan: # Check if execution_plan is not None and not empty
            self.logger.info(f"TaskAnalyzer provided an execution plan: {execution_plan}.")
            planned_agent_instances: List[BaseAgent] = []
            plan_valid = True
            for agent_name in execution_plan:
                if agent_name in available_agents:
                    planned_agent_instances.append(available_agents[agent_name])
                else:
                    self.logger.warning(
                        f"Agent '{agent_name}' from execution_plan not found in available_agents. Plan is invalid."
                    )
                    plan_valid = False
                    break # Stop processing this plan

            if plan_valid:
                self.logger.info(
                    f"Execution plan is valid. Selected {len(planned_agent_instances)} agent(s) in planned order: "
                    f"{[agent.get_name() for agent in planned_agent_instances]}"
                )
                return planned_agent_instances
            else:
                # If plan is invalid, should we fall back to suggested_agents or just return empty?
                # For now, let's say an invalid plan means we cannot proceed with this analysis result confidently.
                # The Coordinator might then decide to try a default agent or report an error.
                # Alternatively, it could fall through to suggestion logic.
                # Current subtask implies returning empty if plan is invalid.
                self.logger.warning("Execution plan was invalid (agent not found). Returning no agents.")
                return []

        # If no execution_plan, or if it was empty, proceed with suggested_agents logic
        suggested_agent_names = analysis_result.get("suggested_agents", [])
        self.logger.info(f"No valid execution plan. Processing suggested_agents: {suggested_agent_names}.")

        selected_agent_instances: List[BaseAgent] = []

        if not available_agents: # Should have been caught by plan logic if plan existed but agents were empty
            self.logger.warning("No agents available for routing (should not happen if plan logic was attempted). Returning empty list.")
            return []

        if suggested_agent_names:
            for agent_name in suggested_agent_names:
                if agent_name in available_agents:
                    selected_agent_instances.append(available_agents[agent_name])
                    self.logger.debug(f"Agent '{agent_name}' added to selection from TaskAnalyzer's suggestions.")
                else:
                    self.logger.warning(
                        f"TaskAnalyzer suggested agent '{agent_name}', but it was not found in available agents. Skipping."
                    )
        else:
            self.logger.info("TaskAnalyzer provided no specific agent suggestions (and no execution plan).")

        if not selected_agent_instances:
            self.logger.info(
                "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). "
                "Considering fallback options."
            )

            if self.config.get("fallback_to_first_available", True):
                if available_agents:
                    first_available_agent_name = list(available_agents.keys())[0]
                    first_agent_instance = available_agents[first_available_agent_name]
                    selected_agent_instances.append(first_agent_instance)
                    self.logger.info(
                        f"Fallback enabled: Selected first available agent: '{first_agent_instance.get_name()}'."
                    )
                else:
                    self.logger.warning("Fallback to first available agent configured, but no agents are available.")
            else:
                self.logger.info("Fallback to first available agent is disabled by configuration.")

        if not selected_agent_instances:
            self.logger.error(
                "Routing engine could not select any agent. No plan, no valid suggestions, and fallback (if any) yielded no agent."
            )
            return []

        selected_names = [agent.get_name() for agent in selected_agent_instances]
        self.logger.info(f"Routing engine finalized selection (suggestion-based). Selected {len(selected_names)} agent(s): {selected_names}")

        return selected_agent_instances


if __name__ == '__main__':
    class MockAgent(BaseAgent):
        def __init__(self, name, capabilities_list=None):
            super().__init__(name, {})
            self._capabilities = {"capabilities": capabilities_list if capabilities_list else []}
            self.logger = get_logger(f"MockAgent.{name}")

        def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "success", "content": f"Mock response from {self.agent_name}"}

        def get_capabilities(self) -> Dict[str, Any]:
            return self._capabilities

    agent_a = MockAgent("AgentA")
    agent_b = MockAgent("AgentB")
    agent_c = MockAgent("AgentC")

    mock_available_agents = {"AgentA": agent_a, "AgentB": agent_b, "AgentC": agent_c}

    engine_with_fallback = RoutingEngine(config={"fallback_to_first_available": True})
    engine_no_fallback = RoutingEngine(config={"fallback_to_first_available": False})

    print("--- Testing Execution Plan Logic ---")
    # Valid Plan
    analysis_plan_valid = {"execution_plan": ["AgentC", "AgentA"]}
    selected_plan = engine_no_fallback.select_agents(analysis_plan_valid, mock_available_agents)
    print(f"Valid Plan Selected: {[a.get_name() for a in selected_plan]}") # Expected: ['AgentC', 'AgentA']
    assert [a.get_name() for a in selected_plan] == ["AgentC", "AgentA"]

    # Invalid Plan (AgentX not available)
    analysis_plan_invalid = {"execution_plan": ["AgentA", "AgentX"]}
    selected_invalid_plan = engine_no_fallback.select_agents(analysis_plan_invalid, mock_available_agents)
    print(f"Invalid Plan Selected: {[a.get_name() for a in selected_invalid_plan]}") # Expected: []
    assert len(selected_invalid_plan) == 0

    # Empty Plan - should fall through to suggestion logic (which then might use fallback)
    analysis_empty_plan = {"execution_plan": [], "suggested_agents": ["AgentB"]}
    selected_empty_plan = engine_no_fallback.select_agents(analysis_empty_plan, mock_available_agents)
    print(f"Empty Plan (uses suggestions): {[a.get_name() for a in selected_empty_plan]}") # Expected: ['AgentB']
    assert [a.get_name() for a in selected_empty_plan] == ["AgentB"]

    analysis_empty_plan_fb = {"execution_plan": [], "suggested_agents": ["AgentX"]} # Suggestion invalid
    selected_empty_plan_fb = engine_with_fallback.select_agents(analysis_empty_plan_fb, mock_available_agents)
    print(f"Empty Plan (suggestion invalid, uses fallback): {[a.get_name() for a in selected_empty_plan_fb]}") # Expected: ['AgentA']
    assert [a.get_name() for a in selected_empty_plan_fb] == ["AgentA"]


    print("\n--- Testing Suggestion Logic (No Execution Plan) ---")
    analysis_sugg_only = {"suggested_agents": ["AgentB", "AgentC"]}
    selected_sugg = engine_no_fallback.select_agents(analysis_sugg_only, mock_available_agents)
    print(f"Suggestion Only Selected: {[a.get_name() for a in selected_sugg]}") # Expected: ['AgentB', 'AgentC']
    # Order might not be guaranteed if based on dict iteration for available_agents lookup, but here suggestions are ordered
    assert [a.get_name() for a in selected_sugg] == ["AgentB", "AgentC"]

    analysis_sugg_invalid_fb = {"suggested_agents": ["AgentY"]}
    selected_sugg_invalid_fb = engine_with_fallback.select_agents(analysis_sugg_invalid_fb, mock_available_agents)
    print(f"Suggestion Invalid (uses fallback): {[a.get_name() for a in selected_sugg_invalid_fb]}") # Expected: ['AgentA']
    assert [a.get_name() for a in selected_sugg_invalid_fb] == ["AgentA"]

    analysis_no_sugg_fb = {"suggested_agents": []}
    selected_no_sugg_fb = engine_with_fallback.select_agents(analysis_no_sugg_fb, mock_available_agents)
    print(f"No Suggestions (uses fallback): {[a.get_name() for a in selected_no_sugg_fb]}") # Expected: ['AgentA']
    assert [a.get_name() for a in selected_no_sugg_fb] == ["AgentA"]

    analysis_no_sugg_no_fb = {"suggested_agents": []}
    selected_no_sugg_no_fb = engine_no_fallback.select_agents(analysis_no_sugg_no_fb, mock_available_agents)
    print(f"No Suggestions (no fallback): {[a.get_name() for a in selected_no_sugg_no_fb]}") # Expected: []
    assert len(selected_no_sugg_no_fb) == 0

    print("\nRoutingEngine refined demonstration with plans completed.")
