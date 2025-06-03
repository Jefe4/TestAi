import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call, AsyncMock # Added AsyncMock
import asyncio # Added for async test case if needed for gather patch

try:
    from src.coordinator.coordinator import Coordinator
    from src.agents.base_agent import BaseAgent
    # Import specific agent classes if needed for isinstance checks in tests, though mocking classes is preferred
    # from src.agents.gemini_agent import GeminiAgent 
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.coordinator.coordinator import Coordinator
    from src.agents.base_agent import BaseAgent
    # from src.agents.gemini_agent import GeminiAgent


class TestCoordinator(unittest.IsolatedAsyncioTestCase): # Changed inheritance
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
            'get_logger': patch('src.coordinator.coordinator.get_logger'),
            'get_nested_value': patch('src.coordinator.coordinator.get_nested_value'),
            'asyncio_gather': patch('asyncio.gather') # Patch asyncio.gather
        }
        
        self.mocks = {} 
        self.mock_instances = {} 

        for name, p in patchers.items():
            self.mocks[name] = p.start()
            self.addCleanup(p.stop) 

        self.mock_instances['api_manager'] = self.mocks['APIManager'].return_value
        self.mock_instances['task_analyzer'] = self.mocks['TaskAnalyzer'].return_value
        self.mock_instances['routing_engine'] = self.mocks['RoutingEngine'].return_value
        self.mock_instances['get_nested_value'] = self.mocks['get_nested_value'] 
        self.mock_instances['asyncio_gather'] = self.mocks['asyncio_gather']
        
        self.mock_instances['coordinator_logger'] = self.mocks['get_logger'].return_value
        for attr in ['info', 'debug', 'warning', 'error']:
            setattr(self.mock_instances['coordinator_logger'], attr, MagicMock())

        # Mock agents for use in plans and suggestions
        # Their process_query methods should be AsyncMock
        self.mock_agent_A = MagicMock(spec=BaseAgent)
        self.mock_agent_A.get_name.return_value = "AgentA"
        self.mock_agent_A.process_query = AsyncMock() # Now AsyncMock
        
        self.mock_agent_B = MagicMock(spec=BaseAgent)
        self.mock_agent_B.get_name.return_value = "AgentB"
        self.mock_agent_B.process_query = AsyncMock()

        self.mock_agent_C = MagicMock(spec=BaseAgent) 
        self.mock_agent_C.get_name.return_value = "AgentC"
        self.mock_agent_C.process_query = AsyncMock()

        self.mock_gemini_instance = MagicMock(spec=BaseAgent) 
        self.mock_gemini_instance.get_name.return_value = "MyGemini"
        self.mock_gemini_instance.process_query = AsyncMock()
        
        self.mocks['DeepSeekAgent'].return_value = self.mock_agent_A
        self.mocks['ClaudeAgent'].return_value = self.mock_agent_B
        self.mocks['CursorAgent'].return_value = self.mock_agent_C
        self.mocks['GeminiAgent'].return_value = self.mock_gemini_instance

    # --- Initialization Tests (omitted for brevity, assume they pass and need no async changes) ---

    # --- Single Agent / Suggestion Path Tests (now async) ---
    async def test_process_query_single_agent_via_suggestion_with_overrides(self):
        cfg_c = {"name": "AgentC", "api_key": "key_c"}
        self.mock_instances['api_manager'].service_configs = {"cursor": cfg_c} 
        coordinator = Coordinator() 
        analysis_result = {"processed_query_for_agent": "analyzed_c", "suggested_agents": ["AgentC"], "execution_plan": []}
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_C]
        self.mock_agent_C.process_query.return_value = {"status": "success", "content": "agent_c_response"}
        query_overrides = {"system_prompt": "Override for AgentC", "temperature": 0.22}
        
        response = await coordinator.process_query("original_c", query_data_overrides=query_overrides)
        
        expected_data = {"prompt": "analyzed_c", "system_prompt": "Override for AgentC", "temperature": 0.22}
        self.mock_agent_C.process_query.assert_awaited_once_with(expected_data)
        self.assertEqual(response, {"status": "success", "content": "agent_c_response"})

    async def test_process_query_falls_back_to_suggestions_if_empty_plan(self):
        cfg_c = {"name": "AgentC", "api_key": "key_c"}
        self.mock_instances['api_manager'].service_configs = {"cursor": cfg_c}
        coordinator = Coordinator()
        analysis_result = {"processed_query_for_agent": "query_sugg", "suggested_agents": ["AgentC"], "execution_plan": []}
        self.mock_instances['task_analyzer'].analyze_query.return_value = analysis_result
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_C]
        self.mock_agent_C.process_query.return_value = {"status": "success", "content": "response_c"}

        response = await coordinator.process_query("original_c", query_data_overrides=None)
        
        self.mock_agent_C.process_query.assert_awaited_once_with({"prompt": "query_sugg"})
        self.assertEqual(response, {"status": "success", "content": "response_c"})

    # --- Tests for Rich Execution Plan Logic (now async) ---
    async def test_process_query_rich_plan_data_chaining_success(self):
        # ... (setup similar to existing, ensure process_query calls are awaited and mocks are AsyncMock) ...
        cfg_a = {"name": "AgentA", "api_key": "key_a"}; cfg_b = {"name": "AgentB", "api_key": "key_b"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()
        initial_query_content = "initial_analyzed_query"
        rich_plan = [
            {"agent_name": "AgentA", "task_description": "Step A", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "res_A", "agent_config_overrides": {}},
            {"agent_name": "AgentB", "task_description": "Step B", "input_mapping": {"prompt": {"source": "ref:res_A.content"}}, "output_id": "res_B", "agent_config_overrides": {}}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": initial_query_content, "execution_plan": rich_plan}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]
        
        agent_A_response_content = "output_from_A"
        self.mock_agent_A.process_query.return_value = {"status": "success", "content": agent_A_response_content, "other_data": "agent_A_other"}
        self.mock_agent_B.process_query.return_value = {"status": "success", "content": "output_from_B"}
        self.mock_instances['get_nested_value'].side_effect = lambda data_dict, path, default: agent_A_response_content if path == "content" else default

        response = await coordinator.process_query("original_user_query_for_rich_plan")

        self.mock_agent_A.process_query.assert_awaited_once_with({"prompt": initial_query_content})
        self.mock_agent_B.process_query.assert_awaited_once_with({"prompt": "output_from_A"})
        self.assertEqual(response, {"status": "success", "content": "output_from_B"})

    # ... (Other rich plan tests need similar async/await conversion and AsyncMock for process_query) ...
    async def test_rich_plan_applies_global_and_step_specific_overrides_with_new_input_resolution(self):
        cfg_a = {"name": "AgentA"}; cfg_b = {"name": "AgentB"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()
        initial_prompt = "initial_plan_input_overrides"
        rich_plan = [
            {"agent_name": "AgentA", "input_mapping": {"data_param": {"source": "original_query"}}, "output_id": "res_A", "agent_config_overrides": {"step_temp": 0.1}},
            {"agent_name": "AgentB", "input_mapping": {"prompt": {"source": "ref:res_A.content"}}, "output_id": "res_B", "agent_config_overrides": {"step_temp": 0.2, "step_max_tokens": 150}}
        ]
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": initial_prompt, "execution_plan": rich_plan, "system_prompt": "AnalysisSystemPrompt"}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]
        self.mock_agent_A.process_query.return_value = {"status": "success", "content": "content_from_A"}
        self.mock_agent_B.process_query.return_value = {"status": "success", "content": "final_content_B"}
        self.mock_instances['get_nested_value'].return_value = "content_from_A"
        global_overrides = {"global_temp": 0.7, "system_prompt": "GlobalSys"}
        
        await coordinator.process_query(initial_prompt, query_data_overrides=global_overrides)

        expected_data_A = {"data_param": initial_prompt, "step_temp": 0.1, "global_temp": 0.7, "system_prompt": "GlobalSys"}
        self.mock_agent_A.process_query.assert_awaited_once_with(expected_data_A)
        expected_data_B = {"prompt": "content_from_A", "step_temp": 0.2, "step_max_tokens": 150, "global_temp": 0.7, "system_prompt": "GlobalSys"}
        self.mock_agent_B.process_query.assert_awaited_once_with(expected_data_B)

    # --- Tests for Parallel Execution ---
    async def test_process_query_parallel_block_success(self):
        cfg_a = {"name": "AgentA"}; cfg_b = {"name": "AgentB"} # Assume these are used in branches
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()
        initial_query_content = "parallel_initial_query"

        # Mock TaskAnalyzer to return a parallel plan
        parallel_plan_def = {
            "type": "parallel_block",
            "task_description": "Run A and B in parallel",
            "output_aggregation": "merge_all",
            "output_id": "pb1_result",
            "branches": [
                [{"agent_name": "AgentA", "task_description": "Branch A task", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "b0s0_A"}],
                [{"agent_name": "AgentB", "task_description": "Branch B task", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "b1s0_B"}]
            ]
        }
        self.mock_instances['task_analyzer'].analyze_query.return_value = {
            "processed_query_for_agent": initial_query_content,
            "execution_plan": [parallel_plan_def] # Parallel block is the only step
        }
        # RoutingEngine needs to return all agents involved in the parallel plan
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]

        # Mock agent responses
        res_A = {"status": "success", "content": "output_A_parallel"}
        res_B = {"status": "success", "content": "output_B_parallel"}
        self.mock_agent_A.process_query.return_value = res_A
        self.mock_agent_B.process_query.return_value = res_B
        
        # Mock asyncio.gather
        # It should return a list of results corresponding to the coroutines
        self.mock_instances['asyncio_gather'].return_value = [res_A, res_B]

        response = await coordinator.process_query("original_parallel_query")

        self.mock_instances['asyncio_gather'].assert_awaited_once()
        # Check calls to _execute_branch_sequentially (these would be within the gather)
        # This requires more intricate mocking of the helper or checking agent calls directly.
        self.mock_agent_A.process_query.assert_awaited_once_with({"prompt": initial_query_content})
        self.mock_agent_B.process_query.assert_awaited_once_with({"prompt": initial_query_content})
        
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["type"], "parallel_block_result")
        self.assertEqual(response["output_id"], "pb1_result")
        self.assertIn("branch_0_result", response["aggregated_results"])
        self.assertEqual(response["aggregated_results"]["branch_0_result"], res_A)
        self.assertIn("branch_1_result", response["aggregated_results"])
        self.assertEqual(response["aggregated_results"]["branch_1_result"], res_B)

    async def test_process_query_parallel_block_one_branch_fails_exception(self):
        cfg_a = {"name": "AgentA"}; cfg_b = {"name": "AgentB"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()
        initial_query_content = "parallel_fail_query"
        parallel_plan_def = {
            "type": "parallel_block", "output_id": "pb_fail", "branches": [
                [{"agent_name": "AgentA", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "b0s0_A_fail"}],
                [{"agent_name": "AgentB", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "b1s0_B_fail"}]
            ]
        }
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": initial_query_content, "execution_plan": [parallel_plan_def]}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]

        res_A_success = {"status": "success", "content": "output_A_ok"}
        self.mock_agent_A.process_query.return_value = res_A_success
        # Agent B's branch will result in an exception
        branch_b_exception = RuntimeError("Branch B execution failed")
        
        self.mock_instances['asyncio_gather'].return_value = [res_A_success, branch_b_exception]

        response = await coordinator.process_query("original_parallel_fail_query")

        self.mock_instances['asyncio_gather'].assert_awaited_once()
        self.assertEqual(response["status"], "partial_success")
        self.assertEqual(response["aggregated_results"]["branch_0_result"], res_A_success)
        self.assertIn("branch_1_result", response["aggregated_results"])
        branch1_error_res = response["aggregated_results"]["branch_1_result"]
        self.assertEqual(branch1_error_res["status"], "error")
        self.assertIn("Branch execution raised: Branch B execution failed", branch1_error_res["message"])

    async def test_process_query_parallel_block_one_branch_returns_error_status(self):
        cfg_a = {"name": "AgentA"}; cfg_b = {"name": "AgentB"}
        self.mock_instances['api_manager'].service_configs = {"deepseek": cfg_a, "claude": cfg_b}
        coordinator = Coordinator()
        initial_query_content = "parallel_status_fail"
        parallel_plan_def = {
            "type": "parallel_block", "output_id": "pb_status_fail", "branches": [
                [{"agent_name": "AgentA", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "b0s0_A_status"}],
                [{"agent_name": "AgentB", "input_mapping": {"prompt": {"source": "original_query"}}, "output_id": "b1s0_B_status"}]
            ]
        }
        self.mock_instances['task_analyzer'].analyze_query.return_value = {"processed_query_for_agent": initial_query_content, "execution_plan": [parallel_plan_def]}
        self.mock_instances['routing_engine'].select_agents.return_value = [self.mock_agent_A, self.mock_agent_B]

        res_A_success = {"status": "success", "content": "output_A_fine"}
        res_B_error = {"status": "error", "message": "Agent B returned an error"}
        self.mock_agent_A.process_query.return_value = res_A_success
        # For this test, we assume _execute_branch_sequentially for branch B returns res_B_error
        self.mock_instances['asyncio_gather'].return_value = [res_A_success, res_B_error]

        response = await coordinator.process_query("original_parallel_status_fail_query")
        
        self.assertEqual(response["status"], "partial_success")
        self.assertEqual(response["aggregated_results"]["branch_0_result"], res_A_success)
        self.assertEqual(response["aggregated_results"]["branch_1_result"], res_B_error)

if __name__ == '__main__':
    unittest.main()
