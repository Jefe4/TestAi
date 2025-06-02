import unittest
from unittest.mock import MagicMock, patch

# Adjust import path based on test execution context
try:
    from src.coordinator.task_analyzer import TaskAnalyzer
    from src.agents.base_agent import BaseAgent
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.coordinator.task_analyzer import TaskAnalyzer
    from src.agents.base_agent import BaseAgent


class TestTaskAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TaskAnalyzer(config={}) # Pass empty config for now

        # Mock the logger used by TaskAnalyzer to suppress output and allow assertions
        self.logger_patcher = patch('src.coordinator.task_analyzer.get_logger')
        # Get the mock that get_logger returns (which is the logger instance)
        self.mock_logger_instance = self.logger_patcher.start().return_value

        # Create mock agents for testing
        self.mock_agent_coder = MagicMock(spec=BaseAgent)
        self.mock_agent_coder.get_name.return_value = "CoderAgent"

        self.mock_agent_writer = MagicMock(spec=BaseAgent)
        self.mock_agent_writer.get_name.return_value = "WriterAgent"

        self.available_agents = {
            "CoderAgent": self.mock_agent_coder,
            "WriterAgent": self.mock_agent_writer
        }

    def tearDown(self):
        self.logger_patcher.stop()

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer.config)
        self.assertIsNotNone(self.analyzer.logger) # Check if logger is initialized
        # Check if the correct logger was obtained by TaskAnalyzer
        # The patch ensures get_logger was called with "TaskAnalyzer"
        # self.logger_patcher.start().assert_called_with("TaskAnalyzer") # This is tricky, patcher.start() returns the mock.
        # Instead, we can check if our self.mock_logger_instance was called by the init log message
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer initialized with config: {}")


    def test_analyze_query_returns_expected_structure(self):
        query = "Test query about Python code."
        analysis = self.analyzer.analyze_query(query, self.available_agents)

        expected_keys = [
            "original_query", "query_type", "complexity",
            "intent", "suggested_agents", "processed_query_for_agent",
            "requires_human_intervention"
        ]
        for key in expected_keys:
            self.assertIn(key, analysis, f"Key '{key}' missing in analysis result.")

        self.assertEqual(analysis["original_query"], query)
        self.assertEqual(analysis["processed_query_for_agent"], query) # Current placeholder behavior

    def test_analyze_query_suggests_all_available_agents_by_default(self):
        # TaskAnalyzer's current implementation with basic keyword matching
        # still defaults to suggesting all agents if no specific capability matching is done.
        query = "Generic question without specific keywords."
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertCountEqual(analysis["suggested_agents"], list(self.available_agents.keys()),
                                  "Should suggest all available agent names by default based on current TaskAnalyzer logic.")

    def test_analyze_query_code_keyword_detection(self):
        # Based on TaskAnalyzer's implementation detail:
        # if "code" in query.lower() or "python" in query.lower():
        #     query_type = "code_related"
        #     intent = "code_generation_or_analysis"
        query = "How to write a function in python?"
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "code_related")
        self.assertEqual(analysis["intent"], "code_generation_or_analysis")

    def test_analyze_query_qa_keyword_detection(self):
        # Based on TaskAnalyzer's implementation detail:
        # elif "what is" in query.lower() or "who is" in query.lower() or "explain" in query.lower():
        #     query_type = "question_answering"
        #     intent = "information_retrieval"
        query = "What is the capital of France?"
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "question_answering")
        self.assertEqual(analysis["intent"], "information_retrieval")

    def test_analyze_query_unknown_type_for_ambiguous_query(self):
        query = "Tell me something interesting about the sky." # No strong keywords from implementation
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        # Current TaskAnalyzer defaults to "unknown" if no keywords match
        self.assertEqual(analysis["query_type"], "unknown")
        self.assertEqual(analysis["intent"], "unknown")

    def test_analyze_query_with_no_available_agents(self):
        query = "Any query."
        analysis = self.analyzer.analyze_query(query, {}) # No agents available
        self.assertEqual(analysis["suggested_agents"], [])
        self.mock_logger_instance.info.assert_any_call(f"Analyzing query: '{query[:100]}...' with 0 agents available.")


if __name__ == '__main__':
    unittest.main()
