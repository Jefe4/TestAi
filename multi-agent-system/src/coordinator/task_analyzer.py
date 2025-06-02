# src/coordinator/task_analyzer.py
"""
Analyzes incoming queries to determine their nature and suggest suitable agents or predefined plans.
"""

from typing import Dict, Any, List, Optional, Set

try:
    from ..agents.base_agent import BaseAgent
    from ..utils.logger import get_logger
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.logger import get_logger # type: ignore

class TaskAnalyzer:
    """
    Analyzes queries to understand their type, complexity, intent,
    and to determine a sequential plan of agents or suggest agents based on capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TaskAnalyzer.
        """
        self.config = config if config is not None else {}
        self.logger = get_logger("TaskAnalyzer")

        self.keyword_map = {
            "summarize_critique_and_keyword": { # New entry for testing overrides
                "keywords": ["summarize critique and list keywords", "review critique and extract topics"],
                "query_type": "advanced_text_processing",
                "intent": "multi_step_critique_analysis",
                "sequential_plan_template": [
                    {"agent_name": "ClaudeAgent", "task_description": "Summarize input text"},
                    {"agent_name": "DeepSeekAgent", "task_description": "Critique the summary.",
                     "agent_config_overrides": {"temperature": 0.1, "max_tokens": 550}}, # Example override
                    {"agent_name": "GeminiAgent", "task_description": "Extract keywords from the critique."}
                ]
            },
            "code_related": {
                "keywords": ["code", "python", "javascript", "function", "class", "algorithm", "debug", "script", "software"],
                "intent": "coding_help",
                "query_type": "code_generation_analysis",
                "capabilities_needed": ["code_generation", "code_analysis", "complex_reasoning"]
            },
            "web_development": {
                "keywords": ["css", "html", "react", "vue", "angular", "frontend", "website", "ui/ux", "web app"],
                "intent": "web_dev_help",
                "query_type": "web_development_assistance",
                "capabilities_needed": ["web_development", "ui_ux", "frontend_frameworks", "css_styling"]
            },
            "question_answering": {
                "keywords": ["what is", "who is", "explain", "define", "tell me about", "how does"],
                "intent": "information_seeking",
                "query_type": "q_and_a",
                "capabilities_needed": ["q&a", "general_analysis", "text_generation"]
            },
            "summarization_simple": {
                "keywords": ["summarize", "tl;dr", "tldr", "gist of", "overview for"],
                "intent": "summarization_request",
                "query_type": "text_summarization",
                "capabilities_needed": ["summarization", "text_generation"]
            },
            "general_query": {
                "keywords": [],
                "intent": "general_interaction",
                "query_type": "general_text_generation",
                "capabilities_needed": ["text_generation", "general_purpose"]
            }
        }
        self.logger.info(f"TaskAnalyzer initialized with config: {self.config} and keyword map.")

    def analyze_query(self, query: str, available_agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        self.logger.info(f"Analyzing query: '{query[:100]}...' with {len(available_agents)} agents available.")

        lower_query = query.lower()

        matched_entry = self.keyword_map["general_query"]
        determined_query_type = matched_entry["query_type"]
        determined_intent = matched_entry["intent"]

        for type_name, type_info in self.keyword_map.items():
            if type_name == "general_query": continue
            if any(keyword in lower_query for keyword in type_info["keywords"]):
                matched_entry = type_info
                determined_query_type = type_info.get("query_type", type_name)
                determined_intent = type_info["intent"]
                self.logger.debug(f"Query matched type '{type_name}' (as {determined_query_type}) with intent '{determined_intent}'.")
                break

        if determined_query_type == self.keyword_map["general_query"]["query_type"]:
             self.logger.debug(f"Query did not match specific types, defaulting to '{determined_query_type}'.")

        suggested_agent_names: List[str] = []
        execution_plan: List[Dict[str, Any]] = []

        if "sequential_plan_template" in matched_entry:
            plan_template = matched_entry["sequential_plan_template"]
            self.logger.info(f"Found sequential plan template for intent '{determined_intent}': {plan_template}")

            current_execution_plan_steps: List[Dict[str, Any]] = [] # Temp list to build plan before assigning
            for i, step_template in enumerate(plan_template):
                agent_name = step_template["agent_name"]
                task_desc = step_template["task_description"]
                agent_config_overrides = step_template.get("agent_config_overrides", {}) # Get overrides or empty dict

                output_id = f"step{i+1}_{agent_name.lower()}"

                current_step_input_mapping: Dict[str, Dict[str,str]] = {} # Outer dict for input types
                if i == 0: # First step
                    current_step_input_mapping["prompt"] = {"source": "original_query"}
                else: # Subsequent steps
                    # Uses output_id of the previously processed step in current_execution_plan_steps
                    previous_step_output_id = current_execution_plan_steps[i-1]["output_id"]
                    current_step_input_mapping["prompt"] = {"source": f"ref:{previous_step_output_id}.content"}

                step_detail = {
                    "agent_name": agent_name,
                    "task_description": task_desc,
                    "input_mapping": current_step_input_mapping,
                    "output_id": output_id,
                    "agent_config_overrides": agent_config_overrides # Add this to the plan step
                }
                current_execution_plan_steps.append(step_detail)

            execution_plan = current_execution_plan_steps # Assign fully built plan
            suggested_agent_names = []

        elif "capabilities_needed" in matched_entry:
            # ... (capability-based suggestion logic remains the same as before) ...
            capabilities_needed = set(matched_entry["capabilities_needed"])
            self.logger.debug(f"Looking for agents with capabilities: {capabilities_needed}")
            if available_agents:
                for agent_name, agent_instance in available_agents.items():
                    try:
                        agent_caps_dict = agent_instance.get_capabilities()
                        agent_declared_capabilities = set(agent_caps_dict.get("capabilities", []))
                        if not all(isinstance(c, str) for c in agent_declared_capabilities):
                             self.logger.warning(f"Agent '{agent_name}' has malformed 'capabilities'. Skipping.")
                             continue
                        if capabilities_needed.intersection(agent_declared_capabilities):
                            suggested_agent_names.append(agent_name)
                            self.logger.debug(f"Agent '{agent_name}' matches. Declared: {agent_declared_capabilities}")
                    except Exception as e:
                        self.logger.error(f"Error processing capabilities for agent '{agent_name}': {e}", exc_info=True)

                if not suggested_agent_names and determined_query_type != self.keyword_map["general_query"]["query_type"]:
                    self.logger.info(f"No specific agents for '{determined_query_type}'. Trying general fallback.")
                    fallback_caps = set(self.keyword_map["general_query"]["capabilities_needed"])
                    for agent_name, agent_instance in available_agents.items():
                        try:
                            agent_declared_capabilities = set(agent_instance.get_capabilities().get("capabilities", []))
                            if fallback_caps.intersection(agent_declared_capabilities) and agent_name not in suggested_agent_names:
                                suggested_agent_names.append(agent_name)
                                self.logger.debug(f"Agent '{agent_name}' added as general fallback.")
                        except Exception as e:
                             self.logger.error(f"Error during fallback capability check for agent '{agent_name}': {e}", exc_info=True)

            if not suggested_agent_names and available_agents:
                 self.logger.warning(f"No agents matched capabilities for query type '{determined_query_type}'.")

        analysis = {
            "original_query": query,
            "query_type": determined_query_type,
            "complexity": "medium",
            "intent": determined_intent,
            "suggested_agents": suggested_agent_names,
            "execution_plan": execution_plan,
            "processed_query_for_agent": query,
            "requires_human_intervention": False
        }

        self.logger.info(
            f"Analysis complete. Query Type: '{determined_query_type}', Intent: '{determined_intent}', "
            f"Plan steps: {len(execution_plan) if execution_plan else 'N/A'}, Suggested (if no plan): {suggested_agent_names if suggested_agent_names else 'N/A'}"
        )
        return analysis

if __name__ == '__main__':
    class MockAgent(BaseAgent):
        def __init__(self, name, capabilities_data: Dict[str, Any]):
            super().__init__(name, {})
            self._capabilities_data = capabilities_data
            self.logger = get_logger(f"MockAgent.{name}")

        def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "success", "content": f"Mock response from {self.agent_name} for {query_data.get('prompt') or query_data.get('prompt_parts')}"}

        def get_capabilities(self) -> Dict[str, Any]:
            return self._capabilities_data

    analyzer = TaskAnalyzer()

    agents_for_test = {
        "CoderBot": MockAgent("CoderBot", {"description": "Codes", "capabilities": ["code_generation", "code_analysis", "complex_reasoning"]}),
        "WebAppWizard": MockAgent("WebAppWizard", {"description": "Web", "capabilities": ["web_development", "css_styling", "text_generation"]}),
        "InfoSeeker": MockAgent("InfoSeeker", {"description": "Q&A", "capabilities": ["q&a", "general_analysis", "text_generation"]}),
        "ClaudeAgent": MockAgent("ClaudeAgent", {"description":"Summarizer", "capabilities":["summarization", "text_generation"]}),
        "DeepSeekAgent": MockAgent("DeepSeekAgent", {"description":"Critiquer/Keyword Extractor", "capabilities":["code_analysis", "text_generation", "complex_reasoning"]}),
        "GeminiAgent": MockAgent("GeminiAgent", {"description":"Keyword Extractor", "capabilities":["multimodal_input", "text_generation"]}),
        "Chatty": MockAgent("Chatty", {"description": "General chat", "capabilities": ["text_generation", "general_purpose"]})
    }

    print(f"--- TaskAnalyzer Demo with {len(agents_for_test)} Mock Agents ---")
    queries_to_test = [
        "summarize critique and list keywords for this document about AI ethics.", # Should trigger new plan
        "Write a python script to list files in a directory.",
        "Explain the theory of relativity.",
        "Hello, how are you today?",
    ]

    for i, test_query in enumerate(queries_to_test):
        print(f"\n--- Query {i+1}: \"{test_query[:70]}...\" ---")
        analysis_result = analyzer.analyze_query(test_query, agents_for_test)
        print(f"  Original Query: {analysis_result['original_query'][:70]}...")
        print(f"  Determined Query Type: {analysis_result['query_type']}")
        print(f"  Determined Intent: {analysis_result['intent']}")
        if analysis_result.get("execution_plan"):
            print(f"  Execution Plan ({len(analysis_result['execution_plan'])} steps):")
            for step_num, step_detail in enumerate(analysis_result['execution_plan']):
                print(f"    Step {step_num+1}: Agent - {step_detail['agent_name']}, Task - '{step_detail['task_description']}'")
                print(f"      Input Mapping: {step_detail['input_mapping']}")
                print(f"      Output ID: {step_detail['output_id']}")
                print(f"      Agent Config Overrides: {step_detail['agent_config_overrides']}")
        if analysis_result.get("suggested_agents"):
            print(f"  Suggested Agents (capability-based): {analysis_result['suggested_agents']}")

    print("\n--- TaskAnalyzer refined demonstration with structured plans and overrides completed. ---")
