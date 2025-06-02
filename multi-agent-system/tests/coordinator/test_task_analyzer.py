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
        self.analyzer = TaskAnalyzer(config={})

        self.logger_patcher = patch('src.coordinator.task_analyzer.get_logger')
        self.mock_logger_instance = self.logger_patcher.start().return_value

        # Create mock agents with diverse capabilities
        self.mock_agent_coder = MagicMock(spec=BaseAgent)
        self.mock_agent_coder.get_name.return_value = "CoderBot"
        self.mock_agent_coder.get_capabilities.return_value = {"capabilities": ["code_generation", "code_analysis", "complex_reasoning"]}

        self.mock_agent_web = MagicMock(spec=BaseAgent)
        self.mock_agent_web.get_name.return_value = "WebAppWizard"
        self.mock_agent_web.get_capabilities.return_value = {"capabilities": ["web_development", "css_styling", "frontend_frameworks", "text_generation"]}

        self.mock_agent_info = MagicMock(spec=BaseAgent)
        self.mock_agent_info.get_name.return_value = "InfoSeeker"
        self.mock_agent_info.get_capabilities.return_value = {"capabilities": ["q&a", "general_analysis", "text_generation"]}

        self.mock_agent_claude = MagicMock(spec=BaseAgent)
        self.mock_agent_claude.get_name.return_value = "ClaudeAgent"
        self.mock_agent_claude.get_capabilities.return_value = {"capabilities": ["summarization", "text_generation"]}

        self.mock_agent_deepseek = MagicMock(spec=BaseAgent)
        self.mock_agent_deepseek.get_name.return_value = "DeepSeekAgent"
        self.mock_agent_deepseek.get_capabilities.return_value = {"capabilities": ["code_analysis", "text_generation", "complex_reasoning"]}

        self.mock_agent_gemini = MagicMock(spec=BaseAgent)
        self.mock_agent_gemini.get_name.return_value = "GeminiAgent"
        self.mock_agent_gemini.get_capabilities.return_value = {"capabilities": ["multimodal_input", "text_generation"]}

        self.mock_agent_chat = MagicMock(spec=BaseAgent)
        self.mock_agent_chat.get_name.return_value = "Chatty"
        self.mock_agent_chat.get_capabilities.return_value = {"capabilities": ["text_generation", "general_purpose"]}

        self.available_agents = {
            "CoderBot": self.mock_agent_coder,
            "WebAppWizard": self.mock_agent_web,
            "InfoSeeker": self.mock_agent_info,
            "ClaudeAgent": self.mock_agent_claude,
            "DeepSeekAgent": self.mock_agent_deepseek,
            "GeminiAgent": self.mock_agent_gemini,
            "Chatty": self.mock_agent_chat
        }
        # TaskAnalyzer's default keyword_map uses "ClaudeAgent", "DeepSeekAgent", "GeminiAgent" in a plan.

    def tearDown(self):
        self.logger_patcher.stop()

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer.config)
        self.assertIsNotNone(self.analyzer.logger)
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer initialized with config: {} and keyword map.")

    def test_analyze_query_returns_expected_structure(self):
        query = "Test query about Python code."
        analysis = self.analyzer.analyze_query(query, self.available_agents)

        expected_keys = [
            "original_query", "query_type", "complexity",
            "intent", "suggested_agents", "processed_query_for_agent",
            "requires_human_intervention", "execution_plan"
        ]
        for key in expected_keys:
            self.assertIn(key, analysis, f"Key '{key}' missing in analysis result.")

        self.assertEqual(analysis["original_query"], query)
        self.assertEqual(analysis["processed_query_for_agent"], query)

    def test_analyze_query_generates_structured_sequential_plan_with_overrides(self):
        query = "summarize critique and list keywords for this document about AI ethics."
        # This query matches "summarize_critique_and_keyword" in TaskAnalyzer's default keyword_map
        # Plan template:
        # 1. ClaudeAgent: Summarize
        # 2. DeepSeekAgent: Critique the summary (with temp 0.1, max_tokens 550)
        # 3. GeminiAgent: Extract keywords from critique

        analysis = self.analyzer.analyze_query(query, self.available_agents)

        self.assertEqual(analysis["query_type"], "advanced_text_processing")
        self.assertEqual(analysis["intent"], "multi_step_critique_analysis")
        self.assertEqual(analysis["suggested_agents"], [])

        self.assertIsInstance(analysis["execution_plan"], list)
        self.assertEqual(len(analysis["execution_plan"]), 3)

        # Step 1: ClaudeAgent
        step1 = analysis["execution_plan"][0]
        self.assertEqual(step1["agent_name"], "ClaudeAgent")
        self.assertEqual(step1["task_description"], "Summarize input text")
        self.assertEqual(step1["input_mapping"], {"prompt": {"source": "original_query"}})
        self.assertEqual(step1["output_id"], "step1_claudeagent")
        self.assertEqual(step1["agent_config_overrides"], {}) # No overrides in template for this step

        # Step 2: DeepSeekAgent
        step2 = analysis["execution_plan"][1]
        self.assertEqual(step2["agent_name"], "DeepSeekAgent")
        self.assertEqual(step2["task_description"], "Critique the summary.")
        self.assertEqual(step2["input_mapping"], {"prompt": {"source": "ref:step1_claudeagent.content"}}) # Referencing previous output_id
        self.assertEqual(step2["output_id"], "step2_deepseekagent")
        self.assertEqual(step2["agent_config_overrides"], {"temperature": 0.1, "max_tokens": 550})

        # Step 3: GeminiAgent
        step3 = analysis["execution_plan"][2]
        self.assertEqual(step3["agent_name"], "GeminiAgent")
        self.assertEqual(step3["task_description"], "Extract keywords from the critique.")
        self.assertEqual(step3["input_mapping"], {"prompt": {"source": "ref:step2_deepseekagent.content"}})
        self.assertEqual(step3["output_id"], "step3_geminiagent")
        self.assertEqual(step3["agent_config_overrides"], {})


    def test_analyze_query_falls_back_to_capability_suggestion_if_no_plan(self):
        query = "Write a python function."
        analysis = self.analyzer.analyze_query(query, self.available_agents)

        self.assertEqual(analysis["query_type"], "code_generation_analysis")
        self.assertEqual(analysis["intent"], "coding_help")
        self.assertEqual(analysis["execution_plan"], [])
        self.assertCountEqual(analysis["suggested_agents"], ["CoderBot"])


    def test_analyze_query_code_keyword_detection_suggests_correct_agent(self):
        query = "How to write a class in javascript?"
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "code_generation_analysis")
        self.assertEqual(analysis["intent"], "coding_help")
        self.assertEqual(analysis["execution_plan"], [])
        self.assertCountEqual(analysis["suggested_agents"], ["CoderBot"])


    def test_analyze_query_qa_keyword_detection_suggests_correct_agent(self):
        query = "Explain what an API is."
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "q_and_a")
        self.assertEqual(analysis["intent"], "information_seeking")
        self.assertEqual(analysis["execution_plan"], [])
        # Expected: InfoSeeker (q&a, general_analysis, text_generation)
        #           Chatty (text_generation, general_purpose)
        #           ClaudeAgent (summarization, text_generation)
        #           DeepSeekAgent (code_analysis, text_generation, complex_reasoning)
        #           WebAppWizard (web_development, css_styling, text_generation)
        #           GeminiAgent (multimodal_input, text_generation)
        # All of these have 'text_generation' or 'general_analysis' which are in "q_and_a" capabilities_needed
        self.assertCountEqual(analysis["suggested_agents"], ["InfoSeeker", "Chatty", "ClaudeAgent", "DeepSeekAgent", "WebAppWizard", "GeminiAgent"])


    def test_analyze_query_general_query_type_suggests_general_agents(self):
        query = "Tell me a joke."
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "general_text_generation")
        self.assertEqual(analysis["intent"], "general_interaction")
        self.assertEqual(analysis["execution_plan"], [])
        # general_query needs: ["text_generation", "general_purpose"]
        # All agents except CoderBot have text_generation or general_purpose
        expected_agents = ["WebAppWizard", "InfoSeeker", "ClaudeAgent", "DeepSeekAgent", "GeminiAgent", "Chatty"]
        self.assertCountEqual(analysis["suggested_agents"], expected_agents)
        self.assertNotIn("CoderBot", analysis["suggested_agents"])

    def test_analyze_query_with_no_available_agents(self):
        query = "Any query."
        analysis = self.analyzer.analyze_query(query, {})
        self.assertEqual(analysis["suggested_agents"], [])
        self.assertEqual(analysis["execution_plan"], [])
        self.mock_logger_instance.info.assert_any_call(f"Analyzing query: '{query[:100]}...' with 0 agents available.")

    def test_analyze_query_plan_with_only_step_specific_configs(self):
        # Modify keyword_map for this specific test to have a single step plan with overrides
        original_keyword_map_entry = self.analyzer.keyword_map.get("single_step_plan_test")
        self.analyzer.keyword_map["single_step_plan_test"] = {
            "keywords": ["single step test with config"],
            "query_type": "single_step_custom_config",
            "intent": "test_single_override",
            "sequential_plan_template": [
                {"agent_name": "ClaudeAgent", "task_description": "Summarize with low temp",
                 "agent_config_overrides": {"temperature": 0.05}}
            ]
        }
        query = "single step test with config please"
        analysis = self.analyzer.analyze_query(query, self.available_agents)

        self.assertEqual(analysis["query_type"], "single_step_custom_config")
        self.assertEqual(len(analysis["execution_plan"]), 1)
        step1 = analysis["execution_plan"][0]
        self.assertEqual(step1["agent_name"], "ClaudeAgent")
        self.assertEqual(step1["input_mapping"], {"prompt": {"source": "original_query"}})
        self.assertEqual(step1["output_id"], "step1_claudeagent")
        self.assertEqual(step1["agent_config_overrides"], {"temperature": 0.05})

        # Restore original or remove if it wasn't there
        if original_keyword_map_entry:
            self.analyzer.keyword_map["single_step_plan_test"] = original_keyword_map_entry
        else:
            del self.analyzer.keyword_map["single_step_plan_test"]


if __name__ == '__main__':
    unittest.main()
