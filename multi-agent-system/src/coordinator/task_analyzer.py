# src/coordinator/task_analyzer.py
"""
Analyzes incoming queries to determine their nature and suggest suitable agents.
"""

from typing import Dict, Any, List, Optional

try:
    from ..agents.base_agent import BaseAgent
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for scenarios where the module might be run directly for testing
    # or if the PYTHONPATH is not set up correctly during development.
    import sys
    import os
    # Navigate up two levels to the 'src' directory, then to project root for utils/agents
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.logger import get_logger # type: ignore


class TaskAnalyzer:
    """
    Analyzes queries to understand their type, complexity, intent,
    and to determine which agents might be best suited to handle them.

    This is a basic implementation and can be expanded with more sophisticated
    analysis techniques (e.g., NLP, keyword extraction, intent recognition models).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TaskAnalyzer.

        Args:
            config: Optional configuration dictionary for the analyzer.
                    Could be used for loading models, setting thresholds, etc.
        """
        self.config = config if config is not None else {}
        self.logger = get_logger("TaskAnalyzer") # Using class name for the logger instance
        self.logger.info(f"TaskAnalyzer initialized with config: {self.config}")

    def analyze_query(self, query: str, available_agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """
        Analyzes the given query against the capabilities of available agents.

        Args:
            query: The user's query string.
            available_agents: A dictionary of available agent instances,
                              keyed by their names. The BaseAgent instances
                              can be inspected for their capabilities.

        Returns:
            A dictionary containing the analysis of the query, including
            a suggestion of which agents might be suitable.
        """
        self.logger.info(f"Analyzing query: '{query[:100]}...' with {len(available_agents)} agents available.")

        # Placeholder analysis logic.
        # In a real system, this would involve more complex logic:
        # - Keyword extraction from the query.
        # - Intent recognition (e.g., "code_generation", "question_answering", "data_analysis").
        # - Matching query intent with agent capabilities (obtained via agent.get_capabilities()).
        # - Complexity assessment (simple, medium, complex) which might influence agent choice or workflow.
        # - Potentially breaking down a complex query into sub-tasks.

        # For now, return a dummy analysis.
        query_type = "unknown"
        complexity = "medium"
        intent = "unknown"

        # Basic keyword matching example (can be significantly improved)
        if "code" in query.lower() or "python" in query.lower() or "javascript" in query.lower():
            query_type = "code_related"
            intent = "code_generation_or_analysis"
        elif "what is" in query.lower() or "who is" in query.lower() or "explain" in query.lower():
            query_type = "question_answering"
            intent = "information_retrieval"
        elif "image" in query.lower() or "picture" in query.lower():
            query_type = "multimodal_query" # If we had multimodal agents
            intent = "image_processing_or_generation"


        # Suggest all agents for now, as we don't have capability matching yet.
        # A more advanced version would filter agents based on `agent.get_capabilities()`.
        suggested_agents_names: List[str] = []
        if available_agents: # Ensure there are agents to suggest
            suggested_agents_names = list(available_agents.keys())

        # If we had capability matching:
        # for agent_name, agent_instance in available_agents.items():
        #     capabilities = agent_instance.get_capabilities().get("capabilities", [])
        #     if intent in capabilities or query_type in capabilities: # Simplistic match
        #         suggested_agents_names.append(agent_name)
        # if not suggested_agents_names and available_agents: # Fallback if no specific match
        #    suggested_agents_names = list(available_agents.keys())


        analysis = {
            "original_query": query,
            "query_type": query_type,
            "complexity": complexity,  # Placeholder
            "intent": intent,
            "suggested_agents": suggested_agents_names,
            "processed_query_for_agent": query, # Placeholder; could be modified/enhanced query
            "requires_human_intervention": False # Placeholder
        }

        self.logger.info(f"Analysis complete. Suggested agents: {analysis['suggested_agents']}. Intent: {analysis['intent']}")
        return analysis

if __name__ == '__main__':
    # Example Usage (basic test)

    # Mock BaseAgent and its get_capabilities for testing purposes
    class MockAgent(BaseAgent):
        def __init__(self, name, capabilities_list=None):
            super().__init__(name, {}) # No real config needed for this mock
            self._capabilities = {"capabilities": capabilities_list if capabilities_list else []}
            self.logger = get_logger(f"MockAgent.{name}") # type: ignore

        def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
            return {"status": "success", "content": f"Mock response from {self.agent_name} for {query_data.get('prompt')}"}

        def get_capabilities(self) -> Dict[str, Any]:
            return self._capabilities

    analyzer_config = {"analysis_mode": "basic"} # Example config
    task_analyzer = TaskAnalyzer(config=analyzer_config)

    agents_available = {
        "CodeAgent": MockAgent("CodeAgent", ["code_generation_or_analysis", "code_related"]),
        "InfoAgent": MockAgent("InfoAgent", ["information_retrieval", "question_answering"]),
        "GeneralAgent": MockAgent("GeneralAgent", ["text_generation"])
    }

    test_query1 = "Generate a Python script for web scraping."
    analysis1 = task_analyzer.analyze_query(test_query1, agents_available) # type: ignore
    print(f"\nAnalysis for query 1 ('{test_query1}'):")
    for key, value in analysis1.items():
        print(f"  {key}: {value}")
    # Expected: query_type = "code_related", intent = "code_generation_or_analysis"

    test_query2 = "What is the capital of France?"
    analysis2 = task_analyzer.analyze_query(test_query2, agents_available) # type: ignore
    print(f"\nAnalysis for query 2 ('{test_query2}'):")
    for key, value in analysis2.items():
        print(f"  {key}: {value}")
    # Expected: query_type = "question_answering", intent = "information_retrieval"

    test_query3 = "Describe a sunset."
    analysis3 = task_analyzer.analyze_query(test_query3, agents_available) # type: ignore
    print(f"\nAnalysis for query 3 ('{test_query3}'):")
    for key, value in analysis3.items():
        print(f"  {key}: {value}")
    # Expected: query_type = "unknown", intent = "unknown" (or matched by GeneralAgent if capabilities were used)

    # Test with no available agents
    analysis_no_agents = task_analyzer.analyze_query("Any query", {})
    print(f"\nAnalysis for query with no agents:")
    for key, value in analysis_no_agents.items():
        print(f"  {key}: {value}")
    assert analysis_no_agents["suggested_agents"] == []

    print("\nTaskAnalyzer basic demonstration completed.")
