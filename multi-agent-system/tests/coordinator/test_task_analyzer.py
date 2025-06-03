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
        
        self.mock_agent_coder = MagicMock(spec=BaseAgent)
        self.mock_agent_coder.get_name.return_value = "CoderBot"
        self.mock_agent_coder.get_capabilities.return_value = {"capabilities": ["code_generation", "code_analysis", "complex_reasoning"]}
        
        self.mock_agent_claude = MagicMock(spec=BaseAgent)
        self.mock_agent_claude.get_name.return_value = "ClaudeAgent" 
        self.mock_agent_claude.get_capabilities.return_value = {"capabilities": ["summarization", "text_generation", "q&a"]}

        self.mock_agent_deepseek = MagicMock(spec=BaseAgent)
        self.mock_agent_deepseek.get_name.return_value = "DeepSeekAgent" 
        self.mock_agent_deepseek.get_capabilities.return_value = {"capabilities": ["code_analysis", "text_generation", "complex_reasoning", "general_analysis"]}
        
        self.mock_agent_gemini = MagicMock(spec=BaseAgent)
        self.mock_agent_gemini.get_name.return_value = "GeminiAgent"
        self.mock_agent_gemini.get_capabilities.return_value = {"capabilities": ["multimodal_input", "text_generation", "summarization"]}
        
        self.available_agents = {
            "CoderBot": self.mock_agent_coder,
            "ClaudeAgent": self.mock_agent_claude,
            "DeepSeekAgent": self.mock_agent_deepseek,
            "GeminiAgent": self.mock_agent_gemini,
        }
        # TaskAnalyzer's default keyword_map uses "ClaudeAgent", "DeepSeekAgent", "GeminiAgent"
        # in both sequential and parallel plan templates.

    def tearDown(self):
        self.logger_patcher.stop()

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer.config)
        self.assertIsNotNone(self.analyzer.logger)
        self.mock_logger_instance.info.assert_any_call("TaskAnalyzer initialized with config: {} and keyword map.")

    def test_analyze_query_returns_expected_structure(self):
        query = "Test query about Python code."
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        expected_keys = ["original_query", "query_type", "complexity", "intent", 
                         "suggested_agents", "execution_plan", "processed_query_for_agent",
                         "requires_human_intervention"]
        for key in expected_keys:
            self.assertIn(key, analysis, f"Key '{key}' missing in analysis result.")

    def test_analyze_query_generates_structured_sequential_plan_with_overrides(self):
        query = "summarize critique and list keywords for this document about AI ethics."
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        
        self.assertEqual(analysis["query_type"], "advanced_text_processing")
        self.assertEqual(analysis["intent"], "multi_step_critique_analysis")
        self.assertEqual(analysis["suggested_agents"], []) 
        
        self.assertIsInstance(analysis["execution_plan"], list)
        # Check if it's not a parallel block structure (i.e., list of steps, not list containing one block dict)
        self.assertTrue(all(isinstance(step, dict) and step.get("type") != "parallel_block" for step in analysis["execution_plan"]))
        self.assertEqual(len(analysis["execution_plan"]), 3)

        # Step 1: ClaudeAgent
        step1 = analysis["execution_plan"][0]
        self.assertEqual(step1["agent_name"], "ClaudeAgent")
        self.assertEqual(step1["task_description"], "Summarize input text")
        self.assertEqual(step1["input_mapping"], {"prompt": {"source": "original_query"}})
        self.assertEqual(step1["output_id"], "sq_summarize_critique_and_keyword_step1_claudeagent")
        self.assertEqual(step1["agent_config_overrides"], {})

        # Step 2: DeepSeekAgent
        step2 = analysis["execution_plan"][1]
        self.assertEqual(step2["agent_name"], "DeepSeekAgent")
        self.assertEqual(step2["task_description"], "Critique the summary.")
        self.assertEqual(step2["input_mapping"], {"prompt": {"source": "ref:sq_summarize_critique_and_keyword_step1_claudeagent.content"}})
        self.assertEqual(step2["output_id"], "sq_summarize_critique_and_keyword_step2_deepseekagent")
        self.assertEqual(step2["agent_config_overrides"], {"temperature": 0.1, "max_tokens": 550})

        # Step 3: GeminiAgent
        step3 = analysis["execution_plan"][2]
        self.assertEqual(step3["agent_name"], "GeminiAgent")
        self.assertEqual(step3["task_description"], "Extract keywords from the critique.")
        self.assertEqual(step3["input_mapping"], {"prompt": {"source": "ref:sq_summarize_critique_and_keyword_step2_deepseekagent.content"}})
        self.assertEqual(step3["output_id"], "sq_summarize_critique_and_keyword_step3_geminiagent")
        self.assertEqual(step3["agent_config_overrides"], {})

    def test_analyze_query_generates_parallel_block_plan(self):
        query = "concurrent market and competitor analysis for new EV startup"
        # This query matches "analyze_market_and_competitors" in TaskAnalyzer's default keyword_map
        
        analysis = self.analyzer.analyze_query(query, self.available_agents)

        self.assertEqual(analysis["query_type"], "parallel_research_analysis")
        self.assertEqual(analysis["intent"], "comprehensive_market_understanding")
        self.assertEqual(analysis["suggested_agents"], []) 
        
        self.assertIsInstance(analysis["execution_plan"], list)
        self.assertEqual(len(analysis["execution_plan"]), 1, "Execution plan should contain one parallel_block dict")
        
        parallel_block = analysis["execution_plan"][0]
        self.assertEqual(parallel_block["type"], "parallel_block")
        self.assertEqual(parallel_block["task_description"], "Run market and competitor analysis concurrently.")
        self.assertEqual(parallel_block["output_aggregation"], "merge_all")
        self.assertTrue(parallel_block["output_id"].startswith("pb_final_analyze_market_and_competitors"))

        self.assertIsInstance(parallel_block["branches"], list)
        self.assertEqual(len(parallel_block["branches"]), 2)

        # Branch 1: DeepSeekAgent for Market Trends
        branch1 = parallel_block["branches"][0]
        self.assertEqual(len(branch1), 1)
        step1_b1 = branch1[0]
        self.assertEqual(step1_b1["agent_name"], "DeepSeekAgent")
        self.assertEqual(step1_b1["task_description"], "Analyze current market trends based on recent reports.")
        self.assertEqual(step1_b1["input_mapping"], {"prompt": {"source": "original_query"}})
        self.assertTrue(step1_b1["output_id"].startswith("pb_analyze_market_and_competitors_b0_step1_deepseekagent"))
        self.assertEqual(step1_b1["agent_config_overrides"], {})

        # Branch 2: ClaudeAgent then GeminiAgent for Competitor Research
        branch2 = parallel_block["branches"][1]
        self.assertEqual(len(branch2), 2)
        
        step1_b2 = branch2[0]
        self.assertEqual(step1_b2["agent_name"], "ClaudeAgent")
        self.assertEqual(step1_b2["task_description"], "Research top 3 competitors, their strengths and weaknesses.")
        self.assertEqual(step1_b2["input_mapping"], {"prompt": {"source": "original_query"}})
        self.assertTrue(step1_b2["output_id"].startswith("pb_analyze_market_and_competitors_b1_step1_claudeagent"))
        self.assertEqual(step1_b2["agent_config_overrides"], {"temperature": 0.65})
        
        step2_b2 = branch2[1]
        self.assertEqual(step2_b2["agent_name"], "GeminiAgent")
        self.assertEqual(step2_b2["task_description"], "Summarize competitor findings into a table.")
        self.assertEqual(step2_b2["input_mapping"], {"prompt": {"source": f"ref:{step1_b2['output_id']}.content"}})
        self.assertTrue(step2_b2["output_id"].startswith("pb_analyze_market_and_competitors_b1_step2_geminiagent"))
        self.assertEqual(step2_b2["agent_config_overrides"], {})


    def test_analyze_query_falls_back_to_capability_suggestion_if_no_plan(self):
        query = "Write a python function." 
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "code_generation_analysis")
        self.assertEqual(analysis["execution_plan"], []) 
        self.assertCountEqual(analysis["suggested_agents"], ["CoderBot"])


    def test_analyze_query_general_query_type_suggests_general_agents(self):
        query = "Tell me a joke." 
        analysis = self.analyzer.analyze_query(query, self.available_agents)
        self.assertEqual(analysis["query_type"], "general_text_generation")
        self.assertEqual(analysis["execution_plan"], [])
        # general_query needs: ["text_generation", "general_purpose"]
        # Based on current setUp: Claude, DeepSeek, Gemini have text_generation. Chatty not in this test's available_agents.
        # CoderBot does not.
        expected_agents = ["ClaudeAgent", "DeepSeekAgent", "GeminiAgent"] # Assuming Chatty isn't in this test's available_agents
        # Recheck available_agents for this test
        current_available_agents = {
            "CoderBot": self.mock_agent_coder, # No match
            "ClaudeAgent": self.mock_agent_claude, # Matches text_generation
            "DeepSeekAgent": self.mock_agent_deepseek, # Matches text_generation
            "GeminiAgent": self.mock_agent_gemini # Matches text_generation
        }
        # Re-evaluating with actual available_agents for this test case
        # self.available_agents in setUp includes CoderBot, ClaudeAgent, DeepSeekAgent, GeminiAgent, Chatty
        # Chatty has general_purpose and text_generation.
        # So, all except CoderBot should be suggested.
        expected_agents_from_setup = ["ClaudeAgent", "DeepSeekAgent", "GeminiAgent", "Chatty"]
        
        # Rerun analysis with the full available_agents from setUp
        analysis_full = self.analyzer.analyze_query(query, self.available_agents)
        self.assertCountEqual(analysis_full["suggested_agents"], expected_agents_from_setup)
        self.assertNotIn("CoderBot", analysis_full["suggested_agents"])


if __name__ == '__main__':
    unittest.main()
