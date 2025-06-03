import unittest
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock 

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


class TestAsyncCoordinatorWorkflows(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.patchers = {}
        self.mocks = {} # To store the mock *classes* or *functions* (e.g., self.mocks['APIManagerClass'])
        self.mock_instances = {} # To store the mock *instances* (e.g., self.mocks['APIManagerInstance'])

        # All paths are relative to where they are imported in 'src.coordinator.coordinator.py'
        coordinator_module_prefix = "src.coordinator.coordinator."
        utils_module_prefix = "src.coordinator.coordinator." # if imported like 'from ..utils.helpers import x'

        dependencies_to_patch = {
            'APIManagerClass': coordinator_module_prefix + 'APIManager',
            'TaskAnalyzerClass': coordinator_module_prefix + 'TaskAnalyzer',
            'RoutingEngineClass': coordinator_module_prefix + 'RoutingEngine',
            'DeepSeekAgentClass': coordinator_module_prefix + 'DeepSeekAgent',
            'ClaudeAgentClass': coordinator_module_prefix + 'ClaudeAgent',
            'CursorAgentClass': coordinator_module_prefix + 'CursorAgent',
            'WindsurfAgentClass': coordinator_module_prefix + 'WindsurfAgent',
            'GeminiAgentClass': coordinator_module_prefix + 'GeminiAgent',
            'get_logger_func': coordinator_module_prefix + 'get_logger',
            'get_nested_value_func': utils_module_prefix + 'get_nested_value'
        }
            
        for name, path_str in dependencies_to_patch.items():
            patcher = patch(path_str)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

        # Configure mock instances returned by the patched class constructors
        self.mock_instances['APIManagerInstance'] = self.mocks['APIManagerClass'].return_value
        self.mock_instances['TaskAnalyzerInstance'] = self.mocks['TaskAnalyzerClass'].return_value
        self.mock_instances['RoutingEngineInstance'] = self.mocks['RoutingEngineClass'].return_value
        
        self.mock_instances['CoordinatorLogger'] = self.mocks['get_logger_func'].return_value
        for attr in ['info', 'debug', 'warning', 'error']:
            setattr(self.mock_instances['CoordinatorLogger'], attr, MagicMock())
        
        self.mock_instances['get_nested_value_func'] = self.mocks['get_nested_value_func'] # This is the function itself

        # Configure mock agent instances
        self.agent_mock_instances = {} 
        # These keys ("deepseek", "claude", etc.) must match the keys used in
        # Coordinator's self.agent_classes AND in APIManager's service_configs for instantiation.
        agent_setup_map = {
            "deepseek": (self.mocks['DeepSeekAgentClass'], "DeepSeekTestAgent"),
            "claude":   (self.mocks['ClaudeAgentClass'],   "ClaudeTestAgent"),
            "cursor":   (self.mocks['CursorAgentClass'],   "CursorTestAgent"),
            "windsurf": (self.mocks['WindsurfAgentClass'], "WindsurfTestAgent"),
            "gemini":   (self.mocks['GeminiAgentClass'],   "GeminiTestAgent")
        }

        for config_key, (mock_class, agent_name_val) in agent_setup_map.items():
            mock_agent_inst = MagicMock(spec=BaseAgent)
            mock_agent_inst.get_name = MagicMock(return_value=agent_name_val)
            mock_agent_inst.process_query = AsyncMock(
                return_value={"status": "success", "content": f"Default mock response from {agent_name_val}"}
            )
            mock_class.return_value = mock_agent_inst
            self.agent_mock_instances[config_key] = mock_agent_inst # Store by config_key for easy access

        # Default behavior for APIManager instance (used by Coordinator)
        # This will be overridden in specific tests if agents need to be instantiated.
        self.mock_instances['APIManagerInstance'].service_configs = {} 
        self.mock_instances['APIManagerInstance'].make_request = AsyncMock(
            return_value={"status": "error", "message": "APIManager.make_request not configured for this test"}
        )

        # Default behavior for TaskAnalyzer instance
        self.mock_instances['TaskAnalyzerInstance'].analyze_query = MagicMock(return_value={
            "original_query": "", "processed_query_for_agent": "", "query_type": "unknown", 
            "intent": "unknown", "execution_plan": [], "suggested_agents": []
        })

        # Default behavior for RoutingEngine instance
        self.mock_instances['RoutingEngineInstance'].select_agents = MagicMock(return_value=[])
            
        # Default behavior for get_nested_value mock function
        self.mocks['get_nested_value_func'].return_value = None 
            
        # Instantiate the real Coordinator. It will use all the mocked dependencies.
        # We provide agent_config_path=None so APIManager uses its default, but since APIManager
        # itself is mocked, its service_configs will be what we set on the mock_instance.
        self.coordinator = Coordinator(agent_config_path=None) 

    async def test_coordinator_initialization_and_basic_mocks(self):
        # Test that Coordinator initializes and its components are the mocks we set up
        self.mocks['APIManagerClass'].assert_called_once() 
        self.mocks['TaskAnalyzerClass'].assert_called_once()
        self.mocks['RoutingEngineClass'].assert_called_once()
        self.mocks['get_logger_func'].assert_any_call("Coordinator") # Coordinator calls get_logger("Coordinator")
        
        self.assertIs(self.coordinator.api_manager, self.mock_instances['APIManagerInstance'])
        self.assertIs(self.coordinator.task_analyzer, self.mock_instances['TaskAnalyzerInstance'])
        self.assertIs(self.coordinator.routing_engine, self.mock_instances['RoutingEngineInstance'])
        self.assertIs(self.coordinator.logger, self.mock_instances['CoordinatorLogger'])
        
        # Test _instantiate_agents part
        # Set service_configs on the *instance* of APIManager that Coordinator uses
        self.mock_instances['APIManagerInstance'].service_configs = {
            "deepseek": {"name": "DeepSeekTestAgent", "api_key": "key1"}, # Matches agent_setup_map
            "claude": {"name": "ClaudeTestAgent", "api_key": "key2"}
        }
        # Re-initialize coordinator to pick up the new service_configs for _instantiate_agents
        # Or, if _instantiate_agents was public, call it. Since it's private, re-init is cleaner for testing init.
        # For this specific test, we'll create a new Coordinator instance.
        
        # Reset call counts for class mocks before re-instantiating for this specific check
        self.mocks['DeepSeekAgentClass'].reset_mock()
        self.mocks['ClaudeAgentClass'].reset_mock()

        coordinator_with_agents = Coordinator(agent_config_path=None)
        
        self.mocks['DeepSeekAgentClass'].assert_called_once_with(
            agent_name="DeepSeekTestAgent", 
            api_manager=self.mock_instances['APIManagerInstance'], 
            config={"name": "DeepSeekTestAgent", "api_key": "key1"}
        )
        self.mocks['ClaudeAgentClass'].assert_called_once_with(
            agent_name="ClaudeTestAgent", 
            api_manager=self.mock_instances['APIManagerInstance'], 
            config={"name": "ClaudeTestAgent", "api_key": "key2"}
        )
        self.assertIn("DeepSeekTestAgent", coordinator_with_agents.agents)
        self.assertIs(coordinator_with_agents.agents["DeepSeekTestAgent"], self.agent_mock_instances["deepseek"])
        self.assertIn("ClaudeTestAgent", coordinator_with_agents.agents)
        self.assertIs(coordinator_with_agents.agents["ClaudeTestAgent"], self.agent_mock_instances["claude"])

    # Placeholder for future integration tests
    async def test_sample_workflow_placeholder(self):
        # This test will be expanded in later steps
        # For now, just a basic assertion to ensure the async test runs
        self.assertTrue(True)
        # Example of awaiting a mocked process_query
        # self.mock_instances['APIManagerInstance'].service_configs = {
        #     "deepseek": {"name": "DeepSeekTestAgent", "api_key": "key1"}
        # }
        # coordinator = Coordinator()
        # self.mock_instances['TaskAnalyzerInstance'].analyze_query.return_value = {
        #     "processed_query_for_agent": "test", "suggested_agents": ["DeepSeekTestAgent"]
        # }
        # self.mock_instances['RoutingEngineInstance'].select_agents.return_value = [self.agent_mock_instances["deepseek"]]
        # response = await coordinator.process_query("hello")
        # self.agent_mock_instances["deepseek"].process_query.assert_awaited_once_with({"prompt":"test"})
        # self.assertEqual(response["content"], "Default mock response from DeepSeekTestAgent")


if __name__ == '__main__':
    unittest.main()
