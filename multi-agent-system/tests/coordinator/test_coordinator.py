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

        # Mock agents for use in plans and suggestions
        self.mock_agent_A = MagicMock(spec=BaseAgent)
        self.mock_agent_A.get_name.return_value = "AgentA"

        self.mock_agent_B = MagicMock(spec=BaseAgent)
        self.mock_agent_B.get_name.return_value = "AgentB"

        self.mock_agent_C = MagicMock(spec=BaseAgent)
        self.mock_agent_C.get_name.return_value = "AgentC"

        self.mock_gemini_instance = MagicMock(spec=BaseAgent)
        self.mock_gemini_instance.get_name.return_value = "MyGemini"

        # Configure mocked agent classes to return these specific instances
        # This is how Coordinator's _instantiate_agents will "create" our mocks
        # if their keys ("deepseek", "claude", "gemini") are in APIManager's service_configs
        self.mocks['DeepSeekAgent'].return_value = self.mock_agent_A # "AgentA" is a DeepSeekAgent
        self.mocks['ClaudeAgent'].return_value = self.mock_agent_B   # "AgentB" is a ClaudeAgent
        self.mocks['CursorAgent'].return_value = self.mock_agent_C   # "AgentC" is a CursorAgent
        self.mocks['GeminiAgent'].return_value = self.mock_gemini_instance


    # --- Initialization Tests ---
    def test_initialization_instantiates_agents_from_config(self):
        cfg_a = {"name": "AgentA", "api_key": "key_a"}
        cfg_b = {"name": "AgentB", "api_key": "key_b"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}

        coordinator = Coordinator()

        self.mocks['DeepSeekAgent'].assert_called_once_with(agent_name="AgentA", api_manager=self.mock_instances['api_manager'], config=cfg_a)
        self.mocks['ClaudeAgent'].assert_called_once_with(agent_name="AgentB", api_manager=self.mock_instances['api_manager'], config=cfg_b)
        self.assertIn("AgentA", coordinator.agents); self.assertIs(coordinator.agents["AgentA"], self.mock_agent_A)
        self.assertIn("AgentB", coordinator.agents); self.assertIs(coordinator.agents["AgentB"], self.mock_agent_B)

    # ... (other init tests: no_configs, skip_unknown, handles_error - assumed to be similar and passing)

    # --- Single Agent / Suggestion Path Tests ---
    def test_process_query_single_agent_via_suggestion_with_overrides(self):
        cfg_c = {"name": "AgentC", "api_key": "key_c"}
        self.mock_instances['api_manager'].service_configs = {"cursor": cfg_c} # AgentC is a CursorAgent
        coordinator = Coordinator()

        analysis_result = {
            "processed_query_for_agent": "analyzed_query_for_agent_c",
            "suggested_agents": ["AgentC"],
            "execution_plan": []
        }
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_C]
        self.mock_agent_C.process_query.return_value = {"status": "success", "content": "agent_c_response"}

        query_overrides = {"system_prompt": "Override for AgentC", "temperature": 0.22}
        response = coordinator.process_query("original_query_for_c", query_data_overrides=query_overrides)

        expected_agent_query_data = {"prompt": "analyzed_query_for_agent_c", "system_prompt": "Override for AgentC", "temperature": 0.22}
        self.mock_agent_C.process_query.assert_called_once_with(expected_agent_query_data)
        self.assertEqual(response, {"status": "success", "content": "agent_c_response"})

    def test_process_query_falls_back_to_suggestions_if_empty_plan(self):
        cfg_c = {"name": "AgentC", "api_key": "key_c"}
        self.mock_instances['api_manager'].service_configs = {"cursor": cfg_c}
        coordinator = Coordinator()

        analysis_result = {"processed_query_for_agent": "query_for_suggestion", "suggested_agents": ["AgentC"], "execution_plan": []}
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_C]
        self.mock_agent_C.process_query.return_value = {"status": "success", "content": "response_from_agent_c"}

        response = coordinator.process_query("original_query_for_agent_c")
        self.mock_agent_C.process_query.assert_called_once_with({"prompt": "query_for_suggestion"})
        self.assertEqual(response, {"status": "success", "content": "response_from_agent_c"})

    # --- Tests for Rich Execution Plan Logic ---
    def test_process_query_rich_plan_driven_sequential_execution_success(self):
        cfg_a = {"name": "AgentA", "api_key": "key_a"}; cfg_b = {"name": "AgentB", "api_key": "key_b"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()

        initial_query_content = "initial_analyzed_query"
        rich_plan = [
            {"agent_name": "AgentA", "task_description": "Step A", "input_mapping": {"prompt_source": "original_query"}, "output_id": "res_A"},
            {"agent_name": "AgentB", "task_description": "Step B", "input_mapping": {"prompt_source": "ref:res_A.content"}, "output_id": "res_B"}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {
            "processed_query_for_agent": initial_query_content,
            "execution_plan": rich_plan
        }
        # RoutingEngine now returns the ordered list of agents based on the plan
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]

        self.mock_agent_A.process_query.return_value = {"status": "success", "content": "output_from_A"}
        self.mock_agent_B.process_query.return_value = {"status": "success", "content": "output_from_B"}

        response = coordinator.process_query("original_user_query_for_rich_plan")

        self.mock_agent_A.process_query.assert_called_once_with({"prompt": initial_query_content})
        self.mock_agent_B.process_query.assert_called_once_with({"prompt": "output_from_A"})
        self.assertEqual(response, {"status": "success", "content": "output_from_B"})
        self.mock_instances['coordinator_logger'].info.assert_any_call(
            "Starting rich sequential execution for 2 agents based on execution_plan: ['AgentA', 'AgentB']"
        )

    def test_rich_plan_fails_if_intermediate_step_fails_status(self):
        cfg_a = {"name": "AgentA", "api_key": "key_a"}; cfg_b = {"name": "AgentB", "api_key": "key_b"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()

        rich_plan_fail = [
            {"agent_name": "AgentA", "task_description": "Step A", "input_mapping": {"prompt_source": "original_query"}, "output_id": "res_A_fail"},
            {"agent_name": "AgentB", "task_description": "Step B", "input_mapping": {"prompt_source": "ref:res_A_fail.content"}, "output_id": "res_B_fail"}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": "input", "execution_plan": rich_plan_fail}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]

        agent_A_error_response = {"status": "error", "message": "AgentA_failed_in_plan"}
        self.mock_agent_A.process_query.return_value = agent_A_error_response

        response = coordinator.process_query("query_for_plan_fail")

        self.mock_agent_A.process_query.assert_called_once()
        self.mock_agent_B.process_query.assert_not_called()
        self.assertEqual(response, agent_A_error_response)
        self.mock_instances['coordinator_logger'].warning.assert_any_call(
            "Agent AgentA in plan step 1 returned status 'error'. Message: AgentA_failed_in_plan. Halting plan execution."
        )

    def test_rich_plan_fails_if_intermediate_step_misses_content(self):
        cfg_a = {"name": "AgentA", "api_key": "key_a"}; cfg_b = {"name": "AgentB", "api_key": "key_b"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()

        rich_plan_no_content = [
            {"agent_name": "AgentA", "task_description": "Step A", "input_mapping": {"prompt_source": "original_query"}, "output_id": "res_A_no_content"},
            {"agent_name": "AgentB", "task_description": "Step B", "input_mapping": {"prompt_source": "ref:res_A_no_content.content"}, "output_id": "res_B_no_content"}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": "input", "execution_plan": rich_plan_no_content}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]
        self.mock_agent_A.process_query.return_value = {"status": "success"} # No "content" field

        response = coordinator.process_query("query_for_plan_no_content")

        self.mock_agent_A.process_query.assert_called_once()
        self.mock_agent_B.process_query.assert_not_called()
        self.assertEqual(response, {"status": "error", "message": "Previous step 'res_A_no_content' missing 'content'."})
        self.mock_instances['coordinator_logger'].error.assert_any_call(
            "Previous step ('res_A_no_content') succeeded but returned no 'content'. Aborting."
        )

    def test_rich_plan_fails_if_first_step_references_previous(self):
        cfg_a = {"name": "AgentA", "api_key": "key_a"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a}
        coordinator = Coordinator()

        invalid_plan_first_step_ref = [
            {"agent_name": "AgentA", "task_description": "Step A", "input_mapping": {"prompt_source": "ref:previous_step.content"}, "output_id": "res_A_invalid"}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": "input", "execution_plan": invalid_plan_first_step_ref}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A]

        response = coordinator.process_query("query_for_invalid_first_step")
        self.assertEqual(response, {"status": "error", "message": "Invalid input_mapping for the first step of the plan."})
        self.mock_agent_A.process_query.assert_not_called()

    def test_rich_plan_with_overrides(self):
        cfg_a = {"name": "AgentA", "api_key": "key_a"}; cfg_b = {"name": "AgentB", "api_key": "key_b"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()

        initial_prompt = "initial_plan_input_overrides"
        rich_plan_overrides = [
            {"agent_name": "AgentA", "task_description": "Step A", "input_mapping": {"prompt_source": "original_query"}, "output_id": "res_A_ovr"},
            {"agent_name": "AgentB", "task_description": "Step B", "input_mapping": {"prompt_source": "ref:res_A_ovr.content"}, "output_id": "res_B_ovr"}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {
            "processed_query_for_agent": initial_prompt,
            "execution_plan": rich_plan_overrides,
            "system_prompt": "System prompt from analysis" # This should apply to first agent if no global override
        }
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]

        self.mock_agent_A.process_query.return_value = {"status": "success", "content": "output_from_A_ovr"}
        self.mock_agent_B.process_query.return_value = {"status": "success", "content": "output_from_B_ovr"}

        # Global override for system_prompt, specific override for temperature
        query_overrides = {"system_prompt": "Global System Prompt For Plan", "temperature": 0.99}
        response = coordinator.process_query("query_rich_plan_overrides", query_data_overrides=query_overrides)

        expected_data_agent_A = {
            "prompt": initial_prompt,
            "system_prompt": "Global System Prompt For Plan", # Overrides analysis_result.system_prompt
            "temperature": 0.99
        }
        self.mock_agent_A.process_query.assert_called_once_with(expected_data_agent_A)

        expected_data_agent_B = {
            "prompt": "output_from_A_ovr",
            "system_prompt": "Global System Prompt For Plan", # Maintained for second step
            "temperature": 0.99
        }
        self.mock_agent_B.process_query.assert_called_once_with(expected_data_agent_B)
        self.assertEqual(response, {"status": "success", "content": "output_from_B_ovr"})


if __name__ == '__main__':
    unittest.main()
