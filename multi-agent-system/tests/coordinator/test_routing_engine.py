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
        # Reset logger mock calls for each test for the primary engine instance
        self.mock_logger_instance.reset_mock()


    def tearDown(self):
        self.logger_patcher.stop()

    def test_initialization(self):
        # Test engine with fallback enabled (default for self.engine_fb_enabled)
        self.assertEqual(self.engine_fb_enabled.config, self.default_config)
        self.assertIsNotNone(self.engine_fb_enabled.logger)
        # Check if get_logger was called for this instance
        # Since get_logger is patched, any call to it will be recorded by self.mock_get_logger
        # The logger instance for self.engine_fb_enabled should be self.mock_logger_instance
        self.mock_get_logger.assert_any_call("RoutingEngine") # Called during init
        self.mock_logger_instance.info.assert_any_call(f"RoutingEngine initialized with config: {self.default_config}")
        self.mock_logger_instance.reset_mock() # Clean for next test part

        # Test engine with fallback disabled
        self.assertEqual(self.engine_fb_disabled.config, self.no_fallback_config)
        self.assertIsNotNone(self.engine_fb_disabled.logger)
        self.mock_logger_instance.info.assert_any_call(f"RoutingEngine initialized with config: {self.no_fallback_config}")


    def test_select_agents_uses_analyzers_suggestions(self):
        analysis = {"suggested_agents": ["AgentB"], "query_type": "specific_type"}
        selected = self.engine_fb_enabled.select_agents(analysis, self.available_agents)

        self.assertEqual(len(selected), 1)
        self.assertIn(self.mock_agent_b, selected)
        self.mock_logger_instance.info.assert_any_call(
            f"Routing engine finalized selection. Selected 1 agent(s): {['AgentB']}"
        )
        # Ensure fallback was not logged as being used
        log_calls_str = " ".join([str(call_args) for call_args, _ in self.mock_logger_instance.info.call_args_list])
        self.assertNotIn("Fallback enabled: Selected first available agent", log_calls_str)


    def test_select_agents_analyzers_suggestions_unavailable_triggers_fallback_when_enabled(self):
        analysis = {"suggested_agents": ["AgentX"], "query_type": "type_x"} # AgentX not available
        selected = self.engine_fb_enabled.select_agents(analysis, self.available_agents)

        self.assertEqual(len(selected), 1)
        self.assertIn(selected[0], self.available_agents.values()) # Should be AgentA (first)
        self.assertEqual(selected[0].get_name(), "AgentA")
        self.mock_logger_instance.warning.assert_any_call(
            "TaskAnalyzer suggested agent 'AgentX', but it was not found in available agents. Skipping."
        )
        self.mock_logger_instance.info.assert_any_call(
            "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). Considering fallback options."
        )
        self.mock_logger_instance.info.assert_any_call(
            f"Fallback enabled: Selected first available agent: 'AgentA'."
        )

    def test_select_agents_analyzers_suggestions_unavailable_no_fallback_when_disabled(self):
        self.mock_logger_instance.reset_mock() # Reset for engine_fb_disabled instance logging
        analysis = {"suggested_agents": ["AgentX"], "query_type": "type_x"}
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents)

        self.assertEqual(len(selected), 0)
        self.mock_logger_instance.warning.assert_any_call(
            "TaskAnalyzer suggested agent 'AgentX', but it was not found in available agents. Skipping."
        )
        self.mock_logger_instance.info.assert_any_call(
             "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). Considering fallback options."
        )
        self.mock_logger_instance.info.assert_any_call("Fallback to first available agent is disabled by configuration.")
        self.mock_logger_instance.error.assert_any_call(
            "Routing engine could not select any agent. No suggestions were processed successfully, and fallback (if any) yielded no agent."
        )


    def test_select_agents_analyzer_returns_empty_suggestions_triggers_fallback_when_enabled(self):
        analysis = {"suggested_agents": [], "query_type": "unknown_type"}
        selected = self.engine_fb_enabled.select_agents(analysis, self.available_agents)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].get_name(), "AgentA") # Fallback to first
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer provided no specific agent suggestions.")
        self.mock_logger_instance.info.assert_any_call(
            "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). Considering fallback options."
        )
        self.mock_logger_instance.info.assert_any_call(
            f"Fallback enabled: Selected first available agent: 'AgentA'."
        )

    def test_select_agents_analyzer_returns_empty_suggestions_no_fallback_when_disabled(self):
        self.mock_logger_instance.reset_mock()
        analysis = {"suggested_agents": [], "query_type": "unknown_type"}
        selected = self.engine_fb_disabled.select_agents(analysis, self.available_agents)

        self.assertEqual(len(selected), 0)
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer provided no specific agent suggestions.")
        self.mock_logger_instance.info.assert_any_call(
            "No agents selected based on TaskAnalyzer's direct suggestions (or suggestions were empty/invalid). Considering fallback options."
        )
        self.mock_logger_instance.info.assert_any_call("Fallback to first available agent is disabled by configuration.")
        self.mock_logger_instance.error.assert_any_call(
             "Routing engine could not select any agent. No suggestions were processed successfully, and fallback (if any) yielded no agent."
        )

    def test_select_agents_no_agents_available_in_system(self):
        analysis = {"suggested_agents": ["AgentA"], "query_type": "test_type"}
        selected = self.engine_fb_enabled.select_agents(analysis, {}) # No agents available

        self.assertEqual(len(selected), 0)
        self.mock_logger_instance.warning.assert_any_call("No agents available for routing. Returning empty list.")

    def test_select_agents_missing_suggestion_key_uses_fallback_when_enabled(self):
        # If 'suggested_agents' key is missing, it defaults to [], triggering fallback
        analysis_key_missing = {"some_other_key": "value", "query_type": "missing_key_analysis"}
        selected = self.engine_fb_enabled.select_agents(analysis_key_missing, self.available_agents)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].get_name(), "AgentA")
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer provided no specific agent suggestions.")
        self.mock_logger_instance.info.assert_any_call(
            f"Fallback enabled: Selected first available agent: 'AgentA'."
        )

if __name__ == '__main__':
    unittest.main()
