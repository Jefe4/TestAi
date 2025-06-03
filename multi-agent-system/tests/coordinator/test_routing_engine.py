import unittest
from unittest.mock import MagicMock, patch

# Adjust import path based on test execution context
try:
    from src.coordinator.routing_engine import RoutingEngine
    from src.agents.base_agent import BaseAgent 
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.coordinator.routing_engine import RoutingEngine
    from src.agents.base_agent import BaseAgent

class TestRoutingEngine(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch('src.coordinator.routing_engine.get_logger')
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger_instance

        self.default_config = {"fallback_to_first_available": True}
        self.engine_fb_enabled = RoutingEngine(config=self.default_config)

        self.no_fallback_config = {"fallback_to_first_available": False}
        self.engine_fb_disabled = RoutingEngine(config=self.no_fallback_config)

        self.mock_agent_a = MagicMock(spec=BaseAgent)
        self.mock_agent_a.get_name.return_value = "AgentA"
        
        self.mock_agent_b = MagicMock(spec=BaseAgent)
        self.mock_agent_b.get_name.return_value = "AgentB"
        
        self.mock_agent_c = MagicMock(spec=BaseAgent)
        self.mock_agent_c.get_name.return_value = "AgentC"
        
        self.available_agents = {
            "AgentA": self.mock_agent_a,
            "AgentB": self.mock_agent_b,
            "AgentC": self.mock_agent_c
        }
        self.mock_logger_instance.reset_mock()

    def tearDown(self):
        self.logger_patcher.stop()

    def test_initialization(self):
        self.assertEqual(self.engine_fb_enabled.config, self.default_config)
        self.assertIsNotNone(self.engine_fb_enabled.logger)
        self.mock_get_logger.assert_any_call("RoutingEngine")
        self.mock_logger_instance.info.assert_any_call(f"RoutingEngine initialized with config: {self.default_config}")
        self.mock_logger_instance.reset_mock() 
        
        self.assertEqual(self.engine_fb_disabled.config, self.no_fallback_config)
        self.assertIsNotNone(self.engine_fb_disabled.logger)
        self.mock_logger_instance.info.assert_any_call(f"RoutingEngine initialized with config: {self.no_fallback_config}")

    # --- Tests for Execution Plan Logic ---
    def test_select_agents_uses_valid_execution_plan(self):
        analysis = {"execution_plan": ["AgentA", "AgentC"], "query_type": "plan_type"}
        # Using engine_fb_disabled to ensure fallback isn't a factor if plan logic fails
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0], self.mock_agent_a) # Order matters
        self.assertEqual(selected[1], self.mock_agent_c)
        self.mock_logger_instance.info.assert_any_call(
            f"Execution plan is valid. Selected 2 agent(s) in planned order: {['AgentA', 'AgentC']}"
        )

    def test_select_agents_invalid_agent_in_execution_plan_returns_empty(self):
        analysis = {"execution_plan": ["AgentA", "AgentX"], "query_type": "plan_type_fail"} # AgentX not available
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 0)
        self.mock_logger_instance.warning.assert_any_call(
            "Agent 'AgentX' from execution_plan not found in available_agents. Plan is invalid."
        )
        self.mock_logger_instance.warning.assert_any_call(
            "Execution plan was invalid (agent not found). Returning no agents."
        )

    def test_select_agents_empty_execution_plan_uses_suggestion_logic(self):
        analysis = {"execution_plan": [], "suggested_agents": ["AgentB"], "query_type": "empty_plan_type"}
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents) # Fallback disabled
        
        self.assertEqual(len(selected), 1)
        self.assertIn(self.mock_agent_b, selected)
        self.mock_logger_instance.info.assert_any_call(
            "No valid execution plan. Processing suggested_agents: ['AgentB']."
        )
        self.mock_logger_instance.info.assert_any_call(
            f"Routing engine finalized selection (suggestion-based). Selected 1 agent(s): {['AgentB']}"
        )

    def test_select_agents_no_execution_plan_key_uses_suggestion_logic(self):
        analysis = {"suggested_agents": ["AgentC"], "query_type": "no_plan_key_type"} # No 'execution_plan' key
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents) # Fallback disabled
        
        self.assertEqual(len(selected), 1)
        self.assertIn(self.mock_agent_c, selected)
        self.mock_logger_instance.info.assert_any_call(
            "No valid execution plan. Processing suggested_agents: ['AgentC']."
        )

    # --- Tests for Suggestion-Based Logic (execution_plan is None or empty) ---
    def test_select_agents_uses_analyzers_suggestions_if_no_plan(self):
        analysis = {"suggested_agents": ["AgentB"], "execution_plan": None, "query_type": "specific_type"}
        selected = self.engine_fb_enabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 1)
        self.assertIn(self.mock_agent_b, selected)
        self.mock_logger_instance.info.assert_any_call(
            f"Routing engine finalized selection (suggestion-based). Selected 1 agent(s): {['AgentB']}"
        )
        log_calls_str = " ".join([str(call_args) for call_args, _ in self.mock_logger_instance.info.call_args_list])
        self.assertNotIn("Fallback enabled: Selected first available agent", log_calls_str)


    def test_select_agents_analyzers_suggestions_unavailable_triggers_fallback_when_enabled_if_no_plan(self):
        analysis = {"suggested_agents": ["AgentX"], "execution_plan": None, "query_type": "type_x"} 
        selected = self.engine_fb_enabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].get_name(), "AgentA") # Fallback to first
        self.mock_logger_instance.warning.assert_any_call(
            "TaskAnalyzer suggested agent 'AgentX', but it was not found in available agents. Skipping."
        )
        self.mock_logger_instance.info.assert_any_call(
            "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). Considering fallback options."
        )
        self.mock_logger_instance.info.assert_any_call(
            f"Fallback enabled: Selected first available agent: 'AgentA'."
        )

    def test_select_agents_analyzers_suggestions_unavailable_no_fallback_when_disabled_if_no_plan(self):
        self.mock_logger_instance.reset_mock()
        analysis = {"suggested_agents": ["AgentX"], "execution_plan": None, "query_type": "type_x"}
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 0)
        self.mock_logger_instance.error.assert_any_call(
            "Routing engine could not select any agent. No plan, no valid suggestions, and fallback (if any) yielded no agent."
        )

    def test_select_agents_analyzer_returns_empty_suggestions_triggers_fallback_when_enabled_if_no_plan(self):
        analysis = {"suggested_agents": [], "execution_plan": None, "query_type": "unknown_type"}
        selected = self.engine_fb_enabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].get_name(), "AgentA") 
        self.mock_logger_instance.info.assert_any_call(
            f"Fallback enabled: Selected first available agent: 'AgentA'."
        )

    def test_select_agents_analyzer_returns_empty_suggestions_no_fallback_when_disabled_if_no_plan(self):
        self.mock_logger_instance.reset_mock()
        analysis = {"suggested_agents": [], "execution_plan": None, "query_type": "unknown_type"}
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents)
        
        self.assertEqual(len(selected), 0)
        self.mock_logger_instance.error.assert_any_call(
             "Routing engine could not select any agent. No plan, no valid suggestions, and fallback (if any) yielded no agent."
        )

    def test_select_agents_no_agents_available_in_system(self):
        # This test is general, applies whether plan or suggestion logic is hit first
        analysis_with_plan = {"execution_plan": ["AgentA"], "query_type": "test_type"}
        selected_plan = self.engine_fb_enabled.select_agents(analysis_with_plan, {})
        self.assertEqual(len(selected_plan), 0)
        self.mock_logger_instance.warning.assert_any_call("No agents available for routing. Returning empty list.")
        self.mock_logger_instance.reset_mock()

        analysis_with_sugg = {"suggested_agents": ["AgentA"], "query_type": "test_type"}
        selected_sugg = self.engine_fb_enabled.select_agents(analysis_with_sugg, {})
        self.assertEqual(len(selected_sugg), 0)
        self.mock_logger_instance.warning.assert_any_call("No agents available for routing. Returning empty list.")


    def test_select_agents_missing_suggestion_and_plan_keys_uses_fallback_when_enabled(self):
        analysis_missing_keys = {"query_type": "missing_all_guidance"}
        selected = self.engine_fb_enabled.select_agents(analysis_missing_keys, self.available_agents)
        
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].get_name(), "AgentA")
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer provided no specific agent suggestions (and no execution plan).")
        self.mock_logger_instance.info.assert_any_call(
            f"Fallback enabled: Selected first available agent: 'AgentA'."
        )

if __name__ == '__main__':
    unittest.main()
