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

        Args:
            analysis_result: The output from TaskAnalyzer.analyze_query.
                             Expected to have a "suggested_agents" key with a list of agent names.
            available_agents: A dictionary of currently available agent instances,
                              keyed by their names.

        Returns:
            A list of BaseAgent instances deemed suitable for the query.
        """
        query_type = analysis_result.get('query_type', 'N/A')
        suggested_agent_names = analysis_result.get("suggested_agents", [])

        self.logger.info(
            f"Selecting agents for query type '{query_type}'. "
            f"TaskAnalyzer suggested: {suggested_agent_names}. "
            f"Available agents: {list(available_agents.keys())}."
        )

        selected_agent_instances: List[BaseAgent] = []

        if not available_agents:
            self.logger.warning("No agents available for routing. Returning empty list.")
            return []

        # Priority 1: Process agents suggested by TaskAnalyzer
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
            self.logger.info("TaskAnalyzer provided no specific agent suggestions.")

        # Priority 2 (Fallback): If no agents were selected from Priority 1
        if not selected_agent_instances:
            self.logger.info(
                "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). "
                "Considering fallback options."
            )

            if self.config.get("fallback_to_first_available", True):
                if available_agents: # Ensure there are agents to fall back to
                    first_available_agent_name = list(available_agents.keys())[0] # Naive: picks first by dict order
                    first_agent_instance = available_agents[first_available_agent_name]
                    selected_agent_instances.append(first_agent_instance)
                    self.logger.info(
                        f"Fallback enabled: Selected first available agent: '{first_agent_instance.get_name()}'."
                    )
                else:
                    # This case should ideally be caught by the initial `if not available_agents:` check,
                    # but added here for robustness if logic path allows.
                    self.logger.warning("Fallback to first available agent configured, but no agents are available in the system.")
            else:
                self.logger.info("Fallback to first available agent is disabled by configuration.")

        if not selected_agent_instances:
            self.logger.error(
                "Routing engine could not select any agent. No suggestions were processed successfully, and fallback (if any) yielded no agent."
            )
            return []

        selected_names = [agent.get_name() for agent in selected_agent_instances]
        self.logger.info(f"Routing engine finalized selection. Selected {len(selected_names)} agent(s): {selected_names}")

        return selected_agent_instances


if __name__ == '__main__':
    # Example Usage (basic test)

    class MockAgent(BaseAgent):
        def __init__(self, name, capabilities_list=None):
            super().__init__(name, {})
            self._capabilities = {"capabilities": capabilities_list if capabilities_list else []}
            self.logger = get_logger(f"MockAgent.{name}")

        def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "success", "content": f"Mock response from {self.agent_name}"}

        def get_capabilities(self) -> Dict[str, Any]:
            return self._capabilities


    # Available agents in the system
    agent_smith = MockAgent("AgentSmith", ["code_generation"])
    agent_jones = MockAgent("AgentJones", ["text_summarization", "general_purpose"])
    agent_brown = MockAgent("AgentBrown", ["general_qa", "general_purpose"])

    all_agents_map = {
        agent_smith.get_name(): agent_smith,
        agent_jones.get_name(): agent_jones,
        agent_brown.get_name(): agent_brown
    }

    print("\n--- Testing RoutingEngine with Fallback Enabled (default) ---")
    routing_engine_fb_enabled = RoutingEngine(config={"fallback_to_first_available": True})

    # Scenario 1: TaskAnalyzer suggests specific, available agent
    analysis1 = {"suggested_agents": ["AgentJones"]}
    selected1 = routing_engine_fb_enabled.select_agents(analysis1, all_agents_map)
    print(f"Scenario 1 (FB Enabled, Specific Suggestion): Selected {[a.get_name() for a in selected1]}")
    assert len(selected1) == 1 and selected1[0].get_name() == "AgentJones"

    # Scenario 2: TaskAnalyzer suggestions are unavailable, fallback occurs
    analysis2 = {"suggested_agents": ["AgentX", "AgentY"]} # None are available
    selected2 = routing_engine_fb_enabled.select_agents(analysis2, all_agents_map)
    print(f"Scenario 2 (FB Enabled, Suggestions Unavailable): Selected {[a.get_name() for a in selected2]}")
    # Expects first agent from all_agents_map (AgentSmith) due to fallback
    assert len(selected2) == 1 and selected2[0].get_name() == "AgentSmith"

    # Scenario 3: TaskAnalyzer returns empty suggestions, fallback occurs
    analysis3 = {"suggested_agents": []}
    selected3 = routing_engine_fb_enabled.select_agents(analysis3, all_agents_map)
    print(f"Scenario 3 (FB Enabled, Empty Suggestions): Selected {[a.get_name() for a in selected3]}")
    assert len(selected3) == 1 and selected3[0].get_name() == "AgentSmith"

    print("\n--- Testing RoutingEngine with Fallback Disabled ---")
    routing_engine_fb_disabled = RoutingEngine(config={"fallback_to_first_available": False})

    # Scenario 4: TaskAnalyzer suggests specific, available agent (fallback setting irrelevant)
    analysis4 = {"suggested_agents": ["AgentBrown"]}
    selected4 = routing_engine_fb_disabled.select_agents(analysis4, all_agents_map)
    print(f"Scenario 4 (FB Disabled, Specific Suggestion): Selected {[a.get_name() for a in selected4]}")
    assert len(selected4) == 1 and selected4[0].get_name() == "AgentBrown"

    # Scenario 5: TaskAnalyzer suggestions are unavailable, no fallback
    analysis5 = {"suggested_agents": ["AgentX"]}
    selected5 = routing_engine_fb_disabled.select_agents(analysis5, all_agents_map)
    print(f"Scenario 5 (FB Disabled, Suggestions Unavailable): Selected {[a.get_name() for a in selected5]}")
    assert len(selected5) == 0

    # Scenario 6: TaskAnalyzer returns empty suggestions, no fallback
    analysis6 = {"suggested_agents": []}
    selected6 = routing_engine_fb_disabled.select_agents(analysis6, all_agents_map)
    print(f"Scenario 6 (FB Disabled, Empty Suggestions): Selected {[a.get_name() for a in selected6]}")
    assert len(selected6) == 0

    # Scenario 7: No agents available in the system
    selected7 = routing_engine_fb_enabled.select_agents(analysis1, {}) # analysis1 suggests AgentJones
    print(f"Scenario 7 (No agents available): Selected {[a.get_name() for a in selected7]}")
    assert len(selected7) == 0

    print("\nRoutingEngine refined demonstration completed.")
