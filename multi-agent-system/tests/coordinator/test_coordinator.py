import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call

try:
    from src.coordinator.coordinator import Coordinator
    from src.agents.base_agent import BaseAgent
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.coordinator.coordinator import Coordinator
    from src.agents.base_agent import BaseAgent

class TestCoordinator(unittest.TestCase):
    def setUp(self):
        patchers = {
            'APIManager': patch('src.coordinator.coordinator.APIManager'),
            'TaskAnalyzer': patch('src.coordinator.coordinator.TaskAnalyzer'),
            'RoutingEngine': patch('src.coordinator.coordinator.RoutingEngine'),
            'DeepSeekAgent': patch('src.coordinator.coordinator.DeepSeekAgent'),
            'ClaudeAgent': patch('src.coordinator.coordinator.ClaudeAgent'),
            'CursorAgent': patch('src.coordinator.coordinator.CursorAgent'),
            'WindsurfAgent': patch('src.coordinator.coordinator.WindsurfAgent'),
            'GeminiAgent': patch('src.coordinator.coordinator.GeminiAgent'),
            'get_logger': patch('src.coordinator.coordinator.get_logger')
        }

        self.mocks = {}
        self.mock_instances = {}

        for name, p in patchers.items():
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

        self.mock_instances['api_manager'] = self.mocks['APIManager'].return_value
        self.mock_instances['task_analyzer'] = self.mocks['TaskAnalyzer'].return_value
        self.mock_instances['routing_engine'] = self.mocks['RoutingEngine'].return_value

        self.mock_instances['coordinator_logger'] = self.mocks['get_logger'].return_value
        for attr in ['info', 'debug', 'warning', 'error']:
            setattr(self.mock_instances['coordinator_logger'], attr, MagicMock())

        self.mock_ds_instance = MagicMock(spec=BaseAgent)
        self.mock_ds_instance.get_name.return_value = "MyDeepSeek"
        self.mocks['DeepSeekAgent'].return_value = self.mock_ds_instance

        self.mock_claude_instance = MagicMock(spec=BaseAgent)
        self.mock_claude_instance.get_name.return_value = "MyClaude"
        self.mocks['ClaudeAgent'].return_value = self.mock_claude_instance

        self.mock_gemini_instance = MagicMock(spec=BaseAgent)
        self.mock_gemini_instance.get_name.return_value = "MyGemini"
        self.mocks['GeminiAgent'].return_value = self.mock_gemini_instance


    def test_initialization_no_agent_configs(self):
        self.mock_instances['api_manager'].service_configs = {}
        coordinator = Coordinator()
        self.assertEqual(coordinator.agents, {})
        self.mock_instances['coordinator_logger'].warning.assert_any_call(
            "No agent configurations found in APIManager's service_configs. "
            "Cannot instantiate agents. Ensure 'agent_configs.yaml' is correctly "
            "populated and accessible, or environment variables are set for services."
        )

    def test_initialization_instantiates_agents_from_config(self):
        ds_config = {"name": "MyDeepSeek", "api_key": "ds_key", "param": 1}
        claude_config = {"name": "MyClaude", "api_key": "claude_key", "param": 2}
        self.mock_instances['api_manager'].service_configs = {
            "deepseek": ds_config,
            "claude": claude_config
        }

        coordinator = Coordinator()

        self.mocks['DeepSeekAgent'].assert_called_once_with(
            agent_name="MyDeepSeek",
            api_manager=self.mock_instances['api_manager'],
            config=ds_config
        )
        self.mocks['ClaudeAgent'].assert_called_once_with(
            agent_name="MyClaude",
            api_manager=self.mock_instances['api_manager'],
            config=claude_config
        )
        self.assertIn("MyDeepSeek", coordinator.agents)
        self.assertIs(coordinator.agents["MyDeepSeek"], self.mock_ds_instance)
        self.assertIn("MyClaude", coordinator.agents)
        self.assertIs(coordinator.agents["MyClaude"], self.mock_claude_instance)
        self.mock_instances['coordinator_logger'].info.assert_any_call(
            "Coordinator initialized successfully with agents: MyDeepSeek, MyClaude" # Order might vary
        )


    def test_initialization_skips_unknown_agent_key(self):
        self.mock_instances['api_manager'].service_configs = {"unknown_agent": {"name": "Unknown", "api_key": "key"}}
        coordinator = Coordinator()
        self.assertEqual(coordinator.agents, {})
        self.mock_instances['coordinator_logger'].warning.assert_any_call(
            "No agent class mapping found for config key: 'unknown_agent'. Skipping agent instantiation."
        )

    def test_initialization_handles_agent_instantiation_error(self):
        ds_config_err = {"name": "MyDeepSeekErr", "api_key": "key"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": ds_config_err}
        self.mocks['DeepSeekAgent'].side_effect = Exception("DeepSeek Init Error")

        coordinator = Coordinator()

        self.assertEqual(coordinator.agents, {})
        self.mock_instances['coordinator_logger'].error.assert_any_call(
            "Failed to instantiate agent 'MyDeepSeekErr' (key: 'deepseek') using class DeepSeekAgent: DeepSeek Init Error",
            exc_info=True
        )

    def test_register_agent_manually(self):
        self.mock_instances['api_manager'].service_configs = {}
        coordinator = Coordinator()

        mock_manual_agent = MagicMock(spec=BaseAgent)
        mock_manual_agent.get_name.return_value = "ManualAgent"

        coordinator.register_agent(mock_manual_agent)

        self.assertIn("ManualAgent", coordinator.agents)
        self.assertIs(coordinator.agents["ManualAgent"], mock_manual_agent)
        self.mock_instances['coordinator_logger'].info.assert_any_call("Agent 'ManualAgent' registered successfully.")

    def test_process_query_no_agents_available(self):
        self.mock_instances['api_manager'].service_configs = {}
        coordinator = Coordinator()

        response = coordinator.process_query("test query")

        self.assertEqual(response, {"status": "error", "message": "No agents available in the system."})
        self.mock_instances['coordinator_logger'].error.assert_any_call(
            "No agents are registered or instantiated. Cannot process query."
        )

    def test_process_query_successful_flow_single_agent_with_overrides(self):
        ds_config_flow = {"name": "MyDeepSeek", "api_key": "key"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": ds_config_flow}
        coordinator = Coordinator()

        analysis_result = {"processed_query_for_agent": "analyzed_query_for_ds", "suggested_agents": ["MyDeepSeek"]}
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result

        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_ds_instance]

        self.mock_ds_instance.process_query.return_value = {"status": "success", "content": "deepseek_agent_response"}

        query_overrides = {"system_prompt": "Test override prompt", "temperature": 0.123}
        response = coordinator.process_query("original_query_text", query_data_overrides=query_overrides)

        self.mock_instances['task_analyzer'].analyze_query.assert_called_once_with("original_query_text", coordinator.agents)
        self.mock_instances['routing_engine'].select_agents.assert_called_once_with(analysis_result, coordinator.agents)

        expected_agent_query_data = {
            "prompt": "analyzed_query_for_ds",
            "system_prompt": "Test override prompt", # From overrides
            "temperature": 0.123                  # From overrides
        }
        self.mock_ds_instance.process_query.assert_called_once_with(expected_agent_query_data)

        self.assertEqual(response, {"status": "success", "content": "deepseek_agent_response"})
        self.mock_instances['coordinator_logger'].info.assert_any_call(
            "Dispatching query to primary selected agent: MyDeepSeek"
        )
        self.mock_instances['coordinator_logger'].info.assert_any_call(
            f"Applying query_data_overrides: {query_overrides}"
        )


    def test_process_query_no_overrides_uses_analysis_prompt(self):
        ds_config_flow = {"name": "MyDeepSeek", "api_key": "key"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": ds_config_flow}
        coordinator = Coordinator()

        # TaskAnalyzer might also return a system_prompt
        analysis_result = {
            "processed_query_for_agent": "analyzed_query_from_taskanalyzer",
            "suggested_agents": ["MyDeepSeek"],
            "system_prompt": "System prompt from TaskAnalyzer"
        }
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_ds_instance]
        self.mock_ds_instance.process_query.return_value = {"status": "success", "content": "agent_response_no_override"}

        response = coordinator.process_query("original_query_no_override") # No query_data_overrides

        expected_agent_query_data = {
            "prompt": "analyzed_query_from_taskanalyzer",
            "system_prompt": "System prompt from TaskAnalyzer" # This comes from analysis_result
        }
        self.mock_ds_instance.process_query.assert_called_once_with(expected_agent_query_data)
        self.assertEqual(response, {"status": "success", "content": "agent_response_no_override"})
        self.mock_instances['coordinator_logger'].debug.assert_any_call("No query_data_overrides provided.")


    def test_process_query_routing_engine_returns_no_agents(self):
        ds_config_route_fail = {"name": "MyDeepSeek", "api_key": "key"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": ds_config_route_fail}
        coordinator = Coordinator()

        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": "query", "suggested_agents": ["SomeAgent"]}
        self.mock_instances['routing_engine'].select_agents.return_value = []

        response = coordinator.process_query("test query for router fail")

        self.assertEqual(response, {"status": "error", "message": "No suitable agent found for the query."})
        self.mock_instances['coordinator_logger'].warning.assert_any_call(
            "Routing engine did not select any agent for the query."
        )

    def test_process_query_selected_agent_raises_exception(self):
        ds_config_agent_ex = {"name": "MyDeepSeek", "api_key": "key"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": ds_config_agent_ex}
        coordinator = Coordinator()

        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": "query", "suggested_agents": ["MyDeepSeek"]}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_ds_instance]
        self.mock_ds_instance.process_query.side_effect = Exception("Agent Internal Processing Error")

        response = coordinator.process_query("test query for agent exception")

        self.assertIn("status", response)
        self.assertEqual(response["status"], "error")
        self.assertIn("Failed to process query with MyDeepSeek: Agent Internal Processing Error", response["message"])
        self.mock_instances['coordinator_logger'].error.assert_any_call(
            "Error during query processing with agent MyDeepSeek: Agent Internal Processing Error",
            exc_info=True
        )

    def test_process_query_adapts_query_data_for_gemini_agent_with_overrides(self):
        gemini_config = {"name": "MyGemini", "api_key": "gemini_key"}
        self.mock_instances['api_manager'].service_configs = {"gemini": gemini_config}
        self.mocks['GeminiAgent'].return_value = self.mock_gemini_instance
        coordinator = Coordinator()

        self.assertIn("MyGemini", coordinator.agents)
        self.assertIs(coordinator.agents["MyGemini"], self.mock_gemini_instance)

        analysis_result = {"processed_query_for_agent": "gemini_query_text", "suggested_agents": ["MyGemini"]}
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_gemini_instance]

        self.mock_gemini_instance.process_query.return_value = {"status": "success", "content": "gemini_response_text"}

        query_overrides = {"custom_param": "gemini_value", "temperature": 0.98}
        response = coordinator.process_query("original_gemini_query_text", query_data_overrides=query_overrides)

        expected_agent_query_data = {
            "prompt_parts": ["gemini_query_text"], # Base from analysis, adapted for Gemini
            "custom_param": "gemini_value",       # From overrides
            "temperature": 0.98                   # From overrides
        }
        self.mock_gemini_instance.process_query.assert_called_once_with(expected_agent_query_data)
        self.assertEqual(response, {"status": "success", "content": "gemini_response_text"})
        self.mock_instances['coordinator_logger'].info.assert_any_call(
            f"Applying query_data_overrides: {query_overrides}"
        )

    def test_process_query_gemini_override_prompt_parts(self):
        gemini_config = {"name": "MyGemini", "api_key": "gemini_key"}
        self.mock_instances['api_manager'].service_configs = {"gemini": gemini_config}
        self.mocks['GeminiAgent'].return_value = self.mock_gemini_instance
        coordinator = Coordinator()

        analysis_result = {"processed_query_for_agent": "original_analysis_prompt", "suggested_agents": ["MyGemini"]}
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_gemini_instance]
        self.mock_gemini_instance.process_query.return_value = {"status": "success"}

        query_overrides = {"prompt_parts": ["overridden_prompt_part_1", "overridden_prompt_part_2"]}
        coordinator.process_query("query", query_data_overrides=query_overrides)

        expected_agent_query_data = {
            "prompt_parts": ["overridden_prompt_part_1", "overridden_prompt_part_2"]
        }
        self.mock_gemini_instance.process_query.assert_called_once_with(expected_agent_query_data)


if __name__ == '__main__':
    unittest.main()
