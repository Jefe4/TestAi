# src/coordinator/task_analyzer.py
"""
Analyzes incoming queries to determine their nature and suggest suitable agents.
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
    and to determine which agents might be best suited to handle them.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TaskAnalyzer.
        """
        self.config = config if config is not None else {}
        self.logger = get_logger("TaskAnalyzer")

        self.keyword_map = {
            "code_related": {
                "keywords": ["code", "python", "javascript", "function", "class", "algorithm", "debug", "script", "software"],
                "intent": "coding_help",
                "capabilities_needed": ["code_generation", "code_analysis", "complex_reasoning"]
            },
            "web_development": {
                "keywords": ["css", "html", "react", "vue", "angular", "frontend", "website", "ui/ux", "web app"],
                "intent": "web_dev_help",
                "capabilities_needed": ["web_development", "ui_ux", "frontend_frameworks", "css_styling"]
            },
            "question_answering": {
                "keywords": ["what is", "who is", "explain", "define", "tell me about", "how does"],
                "intent": "information_seeking",
                "capabilities_needed": ["q&a", "general_analysis", "text_generation"]
            },
            "summarization": {
                "keywords": ["summarize", "tl;dr", "tldr", "gist", "overview of", "key points"],
                "intent": "summarization_request",
                "capabilities_needed": ["summarization", "text_generation"]
            },
            "general_query": { # Fallback category
                "keywords": [], # No specific keywords, or matches if others don't
                "intent": "general_interaction",
                "capabilities_needed": ["text_generation", "general_purpose"] # Broad capabilities
            }
        }
        self.logger.info(f"TaskAnalyzer initialized with config: {self.config} and keyword map.")

    def analyze_query(self, query: str, available_agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """
        Analyzes the given query to determine its type, intent, and suggest suitable agents
        based on keyword matching and agent capabilities.
        """
        self.logger.info(f"Analyzing query: '{query[:100]}...' with {len(available_agents)} agents available.")

        lower_query = query.lower()

        determined_query_type = "general_query"
        determined_intent = self.keyword_map["general_query"]["intent"]
        capabilities_needed: Set[str] = set(self.keyword_map["general_query"]["capabilities_needed"])

        # Determine query type and intent based on keywords
        # More specific categories should be checked first if keyword lists overlap significantly.
        # For now, first match wins.
        for type_name, type_info in self.keyword_map.items():
            if type_name == "general_query": # Skip general_query in this loop, it's the default
                continue
            if any(keyword in lower_query for keyword in type_info["keywords"]):
                determined_query_type = type_name
                determined_intent = type_info["intent"]
                capabilities_needed = set(type_info["capabilities_needed"])
                self.logger.debug(f"Query matched type '{type_name}' with intent '{determined_intent}'. Needed capabilities: {capabilities_needed}")
                break

        if determined_query_type == "general_query":
             self.logger.debug(f"Query did not match specific types, defaulting to '{determined_query_type}' with intent '{determined_intent}'. Needed capabilities: {capabilities_needed}")


        suggested_agent_names: List[str] = []
        if available_agents:
            for agent_name, agent_instance in available_agents.items():
                try:
                    agent_caps_dict = agent_instance.get_capabilities()
                    # Ensure 'capabilities' key exists and is a list of strings
                    agent_declared_capabilities: Set[str] = set(agent_caps_dict.get("capabilities", []))

                    if not isinstance(agent_declared_capabilities, set) or not all(isinstance(c, str) for c in agent_declared_capabilities):
                         self.logger.warning(f"Agent '{agent_name}' has malformed 'capabilities' (not a list of strings). Skipping.")
                         continue

                    # Check if any of the agent's declared capabilities match any of the needed capabilities
                    if capabilities_needed.intersection(agent_declared_capabilities):
                        suggested_agent_names.append(agent_name)
                        self.logger.debug(f"Agent '{agent_name}' matches needed capabilities. Declared: {agent_declared_capabilities}")

                except Exception as e:
                    self.logger.error(f"Error getting or processing capabilities for agent '{agent_name}': {e}", exc_info=True)

            # Fallback if no specific agents were matched for a non-general query type
            if not suggested_agent_names and determined_query_type != "general_query":
                self.logger.info(f"No specific agents found for '{determined_query_type}'. Trying fallback to general purpose agents.")
                fallback_capabilities_needed = set(self.keyword_map["general_query"]["capabilities_needed"])
                for agent_name, agent_instance in available_agents.items():
                    try:
                        agent_caps_dict = agent_instance.get_capabilities()
                        agent_declared_capabilities = set(agent_caps_dict.get("capabilities", []))
                        if fallback_capabilities_needed.intersection(agent_declared_capabilities):
                            if agent_name not in suggested_agent_names: # Avoid duplicates if already added
                                suggested_agent_names.append(agent_name)
                                self.logger.debug(f"Agent '{agent_name}' added as general fallback.")
                    except Exception as e:
                         self.logger.error(f"Error getting or processing capabilities during fallback for agent '{agent_name}': {e}", exc_info=True)

        # If still no agents, it means either no agents are available or none match even general criteria.
        if not suggested_agent_names and available_agents:
            self.logger.warning(
                f"No agents matched primary or fallback capabilities for query type '{determined_query_type}'. "
                "RoutingEngine might apply a final fallback if configured (e.g., first available)."
            )


        analysis = {
            "original_query": query,
            "query_type": determined_query_type,
            "complexity": "medium",  # Placeholder, could be enhanced
            "intent": determined_intent,
            "suggested_agents": suggested_agent_names,
            "processed_query_for_agent": query, # Placeholder for now
            "requires_human_intervention": False # Placeholder
        }

        self.logger.info(
            f"Analysis complete. Query Type: '{determined_query_type}', Intent: '{determined_intent}', "
            f"Suggested Agents: {suggested_agent_names}"
        )
        return analysis

if __name__ == '__main__':
    # Mock BaseAgent and its get_capabilities for testing purposes
    class MockAgent(BaseAgent):
        def __init__(self, name, capabilities_data: Dict[str, Any]): # capabilities_data is the full dict
            super().__init__(name, {})
            self._capabilities_data = capabilities_data
            self.logger = get_logger(f"MockAgent.{name}") # type: ignore

        def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "success", "content": f"Mock response from {self.agent_name} for {query_data.get('prompt')}"}

        def get_capabilities(self) -> Dict[str, Any]:
            return self._capabilities_data

    analyzer = TaskAnalyzer()

    # Define Mock Agents with varied capabilities
    agents_for_test = {
        "CoderBot": MockAgent("CoderBot", {"description": "Codes things", "capabilities": ["code_generation", "code_analysis", "complex_reasoning"]}),
        "WebAppWizard": MockAgent("WebAppWizard", {"description": "Builds web apps", "capabilities": ["web_development", "css_styling", "frontend_frameworks", "text_generation"]}),
        "InfoSeeker": MockAgent("InfoSeeker", {"description": "Finds info", "capabilities": ["q&a", "general_analysis", "text_generation"]}),
        "SummaryPro": MockAgent("SummaryPro", {"description": "Summarizes text", "capabilities": ["summarization", "text_generation"]}),
        "Chatty": MockAgent("Chatty", {"description": "General chat", "capabilities": ["text_generation", "general_purpose"]})
    }
    print(f"--- TaskAnalyzer Demo with {len(agents_for_test)} Mock Agents ---")

    queries_to_test = [
        "Write a python script to list files in a directory.",
        "What are the best practices for CSS grid?",
        "Explain the theory of relativity.",
        "Give me a summary of this article: [long article text here...]",
        "Hello, how are you today?",
        "Debug this Java code snippet for me.",
        "What is the weather like?", # Should hit general_query or q&a
        "Create an overview of project management steps." # Should hit summarization
    ]

    for i, test_query in enumerate(queries_to_test):
        print(f"\n--- Query {i+1}: \"{test_query[:70]}...\" ---")
        analysis_result = analyzer.analyze_query(test_query, agents_for_test) # type: ignore
        print(f"  Original Query: {analysis_result['original_query'][:70]}...")
        print(f"  Determined Query Type: {analysis_result['query_type']}")
        print(f"  Determined Intent: {analysis_result['intent']}")
        print(f"  Suggested Agents: {analysis_result['suggested_agents']}")

    # Test with no available agents
    print("\n--- Query with No Available Agents ---")
    analysis_no_agents = analyzer.analyze_query("Any query will do.", {})
    print(f"  Suggested Agents (none available): {analysis_no_agents['suggested_agents']}")
    assert analysis_no_agents['suggested_agents'] == []

    print("\n--- TaskAnalyzer refined demonstration completed. ---")
