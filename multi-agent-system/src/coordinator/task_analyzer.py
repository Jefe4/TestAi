# src/coordinator/task_analyzer.py
"""
Analyzes incoming user queries to determine their nature, intent, complexity,
and to generate an execution plan (sequential or parallel) or suggest
suitable agents based on their capabilities.
"""

from typing import Dict, Any, List, Optional, Set # Standard typing modules

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
    and to determine a sequential or parallel plan of agents, or suggest agents based on capabilities.

    The analyzer uses a keyword-based approach to categorize queries and map them to
    predefined plan templates (sequential or parallel) or to a set of capabilities
    for agent suggestion. It also assesses query complexity and recommends an approach.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TaskAnalyzer.

        Args:
            config: Optional configuration dictionary for the TaskAnalyzer.
                    Currently not used but can be extended for future customization.
        """
        self.config = config if config is not None else {}
        self.logger = get_logger("TaskAnalyzer")
        
        # Defines how queries are mapped to plans or capabilities.
        # Each key (e.g., "analyze_market_and_competitors") is a descriptive name for a type of task.
        # - "keywords": List of phrases that, if found in a query, trigger this task type.
        # - "query_type": A specific categorization for this type of query.
        # - "intent": The inferred user intent.
        # - "parallel_plan_template" or "sequential_plan_template": Defines a multi-step execution plan.
        #   - "task_description": Overall description for the plan or block.
        #   - "output_aggregation" (for parallel): How to combine branch outputs (e.g., "merge_all").
        #   - "branches" (for parallel): List of branches, each a list of sequential steps.
        #   - Each step template includes "agent_name", "task_description", and optional "agent_config_overrides".
        # - "capabilities_needed" (if no template): List of capabilities agents should have to handle this query type.
        self.keyword_map = {
            "analyze_market_and_competitors": {
                "keywords": ["concurrent market and competitor analysis", "market research and competitive landscape"],
                "query_type": "parallel_research_analysis",
                "intent": "comprehensive_market_understanding",
                "parallel_plan_template": {
                    "task_description": "Run market and competitor analysis concurrently.",
                    "output_aggregation": "merge_all", # Example aggregation strategy
                    "branches": [
                        [ # Branch 1: Market Trends
                            {"agent_name": "DeepSeekAgent", "task_description": "Analyze current market trends based on recent reports."}
                        ],
                        [ # Branch 2: Competitor Research
                            {"agent_name": "ClaudeAgent", "task_description": "Research top 3 competitors, their strengths and weaknesses.", 
                             "agent_config_overrides": {"temperature": 0.65}},
                            {"agent_name": "GeminiAgent", "task_description": "Summarize competitor findings into a table."}
                        ]
                    ]
                }
            },
            "summarize_critique_and_keyword": { 
                "keywords": ["summarize critique and list keywords", "review critique and extract topics"],
                "query_type": "advanced_text_processing",
                "intent": "multi_step_critique_analysis",
                "sequential_plan_template": [
                    {"agent_name": "ClaudeAgent", "task_description": "Summarize input text"},
                    {"agent_name": "DeepSeekAgent", "task_description": "Critique the summary.", 
                     "agent_config_overrides": {"temperature": 0.1, "max_tokens": 550}}, 
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
        self.logger.info(f"TaskAnalyzer initialized with keyword map for {len(self.keyword_map)} task types.")

    def _process_plan_steps(self, plan_step_templates: List[Dict[str, Any]], plan_id_prefix: str) -> List[Dict[str, Any]]:
        """
        Helper function to convert plan step templates into full step definitions.

        This function takes a list of simplified step templates (typically containing
        agent_name, task_description, and overrides) and enriches them with
        default input mappings and unique output IDs.

        For the first step in a sequence (or branch), input is mapped from "original_query".
        For subsequent steps, input is mapped from the "content" of the "previous_step"'s output.

        Args:
            plan_step_templates: A list of dictionaries, where each dictionary is a template
                                 for a step in an execution plan.
            plan_id_prefix: A string prefix used to generate unique output IDs for steps
                            (e.g., "sq_summarize" or "pb_market_b0").

        Returns:
            A list of processed step definition dictionaries, ready to be included in
            an execution plan.
        """
        processed_steps: List[Dict[str, Any]] = []
        for i, step_template in enumerate(plan_step_templates):
            agent_name = step_template["agent_name"] # Name of the agent to execute this step
            task_desc = step_template["task_description"]
            agent_config_overrides = step_template.get("agent_config_overrides", {})
            
            # Generate a unique output ID for this step, incorporating the prefix, step number, and agent name.
            output_id = f"{plan_id_prefix}_step{i+1}_{agent_name.lower().replace('agent', '')}"
            
            current_step_input_mapping: Dict[str, Dict[str,str]] = {}
            # Default input mapping: first step takes original query, subsequent steps take previous content.
            if i == 0: # First step in the sequence
                current_step_input_mapping["prompt"] = {"source": "original_query"}
            else: # Subsequent steps
                # Reference the 'output_id' of the previously processed step in this sequence.
                previous_step_output_id = processed_steps[i-1]["output_id"]
                current_step_input_mapping["prompt"] = {"source": f"ref:{previous_step_output_id}.content"}
            
            processed_steps.append({
                "agent_name": agent_name,
                "task_description": task_desc,
                "input_mapping": current_step_input_mapping,
                "output_id": output_id,
                "agent_config_overrides": agent_config_overrides
            })
        return processed_steps

    def analyze_query(self, query: str, available_agents: Dict[str, BaseAgent], context_trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyzes a given query to determine its characteristics and formulate an execution strategy.

        This involves:
        1. Matching the query against a keyword map to determine query type and intent.
        2. Assigning a complexity score (simple, moderate, complex) based on query characteristics
           and matched task type.
        3. Generating an execution plan (sequential or parallel) if a predefined template matches the query.
        4. Suggesting suitable agents based on capabilities if no predefined plan is found.
        5. Determining a recommended approach (e.g., "multi_agent_plan", "single_agent_with_tools").

        Args:
            query: The user's query string.
            available_agents: A dictionary of available agent instances, keyed by name.
            context_trace: Optional list of previous interaction events, currently logged but not used
                           for altering analysis logic in this version.

        Returns:
            A dictionary containing the analysis results, including:
            - "original_query": The input query.
            - "query_type": Categorization of the query.
            - "complexity": Assessed complexity ("simple", "moderate", "complex").
            - "intent": Inferred user intent.
            - "suggested_agents": List of agent names if no plan is made.
            - "execution_plan": A list of step definitions (can be a parallel block).
            - "processed_query_for_agent": The query text to be sent to the first agent.
            - "requires_human_intervention": Boolean, currently hardcoded to False.
            - "recommended_approach": String suggesting an overall strategy.
        """
        self.logger.info(f"Analyzing query: '{query[:100]}...' with {len(available_agents)} agents available.")
        if context_trace:
            self.logger.debug(f"Received context_trace with {len(context_trace)} events.")
        # Future: context_trace could be used to influence query understanding or plan generation.
        
        lower_query = query.lower() # For case-insensitive keyword matching
        
        # Default to 'general_query' if no specific keywords match
        matched_entry = self.keyword_map["general_query"]
        matched_type_name = "general_query"

        # Iterate through keyword_map to find a matching task type
        for type_name, type_info in self.keyword_map.items():
            if type_name == "general_query": continue # Skip default, check specific ones first
            if any(keyword in lower_query for keyword in type_info.get("keywords", [])):
                matched_entry = type_info
                matched_type_name = type_name
                self.logger.debug(f"Query matched pre-defined type: '{type_name}'")
                break 
        
        determined_query_type = matched_entry.get("query_type", "unknown")
        determined_intent = matched_entry.get("intent", "unknown")
        self.logger.debug(f"Determined Query Type: '{determined_query_type}', Intent: '{determined_intent}' (based on map key: '{matched_type_name}').")

        # --- Complexity Score Logic ---
        complexity_score = "moderate" # Default complexity
        if "parallel_plan_template" in matched_entry or "sequential_plan_template" in matched_entry:
            complexity_score = "complex" # Predefined multi-step plans are considered complex
        elif matched_type_name == "general_query" or matched_type_name == "question_answering":
            # Simple Q&A or general queries might be simple if short
            if len(lower_query.split()) < 15: # Heuristic for short query
                complexity_score = "simple"

        # Adjust complexity based on specific keywords in the query
        simplifying_keywords = ["briefly", "simple", "one sentence", "quick summary"]
        complexifying_keywords = ["detailed", "comprehensive", "in-depth", "research", "analyze and compare", "step-by-step plan"]

        if any(s_kw in lower_query for s_kw in simplifying_keywords):
            if complexity_score == "moderate": complexity_score = "simple"
            # Note: We don't downgrade "complex" plans to "simple" easily based on just these keywords.
        if any(c_kw in lower_query for c_kw in complexifying_keywords):
            if complexity_score == "simple": complexity_score = "moderate"
            elif complexity_score == "moderate": complexity_score = "complex"

        self.logger.debug(f"Final determined complexity score: {complexity_score}")

        # --- Execution Plan Generation or Agent Suggestion ---
        suggested_agent_names: List[str] = []
        execution_plan: List[Dict[str, Any]] = []
        has_defined_plan = False # Flag to track if a plan template was used

        # Check for and process a parallel plan template
        if "parallel_plan_template" in matched_entry:
            parallel_template = matched_entry["parallel_plan_template"]
            self.logger.info(f"Generating parallel execution plan based on template for intent: '{determined_intent}'.")
            
            processed_branches = []
            for branch_idx, branch_template_steps in enumerate(parallel_template.get("branches", [])):
                branch_id_prefix = f"pb_{matched_type_name}_b{branch_idx}" # Prefix for branch step output IDs
                processed_branch = self._process_plan_steps(branch_template_steps, branch_id_prefix)
                processed_branches.append(processed_branch)
            
            parallel_block_output_id = f"pb_final_{matched_type_name}"
            execution_plan = [{
                "type": "parallel_block",
                "task_description": parallel_template.get("task_description", "Execute tasks in parallel."),
                "output_aggregation": parallel_template.get("output_aggregation", "merge_all"), # How to combine branch results
                "output_id": parallel_block_output_id, # ID for the output of the entire parallel block
                "branches": processed_branches
            }]
            # For plan-based execution, specific agents are defined in the plan, so no general suggestion needed here.
            has_defined_plan = True

        # Check for and process a sequential plan template
        elif "sequential_plan_template" in matched_entry:
            plan_template = matched_entry["sequential_plan_template"]
            self.logger.info(f"Generating sequential execution plan based on template for intent: '{determined_intent}'.")
            seq_id_prefix = f"sq_{matched_type_name}" # Prefix for sequential step output IDs
            execution_plan = self._process_plan_steps(plan_template, seq_id_prefix)
            has_defined_plan = True
        
        # If no predefined plan, suggest agents based on capabilities
        elif "capabilities_needed" in matched_entry:
            capabilities_needed = set(matched_entry["capabilities_needed"])
            self.logger.debug(f"No predefined plan. Looking for agents with capabilities: {capabilities_needed}")
            if available_agents:
                for agent_name, agent_instance in available_agents.items():
                    try:
                        agent_caps_dict = agent_instance.get_capabilities()
                        agent_declared_capabilities = set(agent_caps_dict.get("capabilities", []))
                        # Ensure capabilities are strings before set operations
                        if not all(isinstance(c, str) for c in agent_declared_capabilities):
                             self.logger.warning(f"Agent '{agent_name}' has malformed 'capabilities' (non-string items). Skipping.")
                             continue
                        if capabilities_needed.intersection(agent_declared_capabilities):
                            suggested_agent_names.append(agent_name)
                            self.logger.debug(f"Agent '{agent_name}' matches required capabilities. Declared: {agent_declared_capabilities}")
                    except Exception as e:
                        self.logger.error(f"Error processing capabilities for agent '{agent_name}': {e}", exc_info=True)

                # Fallback to general capabilities if no specific agents found for non-general query types
                if not suggested_agent_names and determined_query_type != self.keyword_map["general_query"]["query_type"]:
                    self.logger.info(f"No specific agents found for '{determined_query_type}'. Trying general purpose agent fallback.")
                    fallback_caps = set(self.keyword_map["general_query"]["capabilities_needed"])
                    for agent_name, agent_instance in available_agents.items():
                        try:
                            agent_declared_capabilities = set(agent_instance.get_capabilities().get("capabilities", []))
                            if fallback_caps.intersection(agent_declared_capabilities) and agent_name not in suggested_agent_names:
                                suggested_agent_names.append(agent_name)
                                self.logger.debug(f"Agent '{agent_name}' added as general fallback.")
                        except Exception as e:
                             self.logger.error(f"Error during fallback capability check for agent '{agent_name}': {e}", exc_info=True)
            
            if not suggested_agent_names and available_agents: # Log if still no agents after all checks
                 self.logger.warning(f"No agents (neither specific nor fallback) matched capabilities for query type '{determined_query_type}'.")


        # --- Recommended Approach Logic ---
        recommended_approach = "multi_agent_plan" # Default assumption, especially if a plan was generated
        if not has_defined_plan and complexity_score == "simple":
            # If no specific plan template was matched (i.e., it's a capability-based suggestion)
            # AND the complexity is assessed as "simple", then suggest a simpler approach.
            recommended_approach = "single_agent_with_tools"
            # "single_agent_with_tools" implies that a single, versatile agent might handle it,
            # possibly by being equipped with appropriate tools in a more advanced system.

        # Construct the final analysis dictionary
        analysis = {
            "original_query": query,
            "query_type": determined_query_type,
            "complexity": complexity_score,
            "intent": determined_intent,
            "suggested_agents": suggested_agent_names, # Empty if a plan is present
            "execution_plan": execution_plan, # Empty if no plan template matched
            "processed_query_for_agent": query, # This could be refined further, e.g., by extracting key info
            "requires_human_intervention": False, # Placeholder for future functionality
            "recommended_approach": recommended_approach
        }

        # Log a summary of the analysis
        self.logger.info(
            f"Analysis complete. Query Type: '{determined_query_type}', Intent: '{determined_intent}', "
            f"Complexity: '{complexity_score}', Recommended Approach: '{recommended_approach}', "
            f"Plan type: {'Parallel' if execution_plan and execution_plan[0].get('type') == 'parallel_block' else 'Sequential' if execution_plan else 'Suggestion'}, "
            f"Plan steps/branches: {len(execution_plan[0]['branches']) if execution_plan and execution_plan[0].get('type') == 'parallel_block' else len(execution_plan) if execution_plan else 'N/A'}, "
            f"Suggested (if no plan): {suggested_agent_names if suggested_agent_names else 'N/A'}"
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
        "ClaudeAgent": MockAgent("ClaudeAgent", {"description":"Summarizer/Competitor Researcher", "capabilities":["summarization", "text_generation", "q&a"]}), 
        "DeepSeekAgent": MockAgent("DeepSeekAgent", {"description":"Market Analyzer/Critiquer", "capabilities":["code_analysis", "text_generation", "complex_reasoning", "general_analysis"]}),
        "GeminiAgent": MockAgent("GeminiAgent", {"description":"Keyword Extractor/Table Summarizer", "capabilities":["multimodal_input", "text_generation", "summarization"]}),
        "Chatty": MockAgent("Chatty", {"description": "General chat", "capabilities": ["text_generation", "general_purpose"]})
    }
    
    print(f"--- TaskAnalyzer Demo with {len(agents_for_test)} Mock Agents ---")
    queries_to_test = [
        "concurrent market and competitor analysis for new EV startup", # Should trigger parallel plan
        "summarize critique and list keywords for this document about AI ethics.", # Should trigger sequential plan
        "Write a python script to list files in a directory.", # Should trigger capability suggestion
        "Explain the theory of relativity.", 
        "Hello, how are you today?", 
    ]

    for i, test_query in enumerate(queries_to_test):
        print(f"\n--- Query {i+1}: \"{test_query[:70]}...\" ---")
        analysis_result = analyzer.analyze_query(test_query, agents_for_test)
        print(f"  Original Query: {analysis_result['original_query'][:70]}...")
        print(f"  Determined Query Type: {analysis_result['query_type']}")
        print(f"  Determined Intent: {analysis_result['intent']}")
        print(f"  Complexity: {analysis_result['complexity']}") # Added print
        print(f"  Recommended Approach: {analysis_result['recommended_approach']}") # Added print
        
        if analysis_result.get("execution_plan"):
            plan = analysis_result["execution_plan"]
            if plan and plan[0].get("type") == "parallel_block":
                pb = plan[0]
                print(f"  Parallel Block Plan: '{pb.get('task_description')}' (Agg: {pb.get('output_aggregation')}, Block ID: {pb.get('output_id')})")
                for b_idx, branch in enumerate(pb.get("branches", [])):
                    print(f"    Branch {b_idx+1} ({len(branch)} steps):")
                    for step_num, step_detail in enumerate(branch):
                        print(f"      Step {step_num+1}: Agent - {step_detail['agent_name']}, Task - '{step_detail['task_description']}'")
                        print(f"        Input Mapping: {step_detail['input_mapping']}")
                        print(f"        Output ID: {step_detail['output_id']}")
                        print(f"        Agent Config Overrides: {step_detail['agent_config_overrides']}")
            else: # Sequential plan
                print(f"  Sequential Execution Plan ({len(plan)} steps):")
                for step_num, step_detail in enumerate(plan):
                    print(f"    Step {step_num+1}: Agent - {step_detail['agent_name']}, Task - '{step_detail['task_description']}'")
                    print(f"      Input Mapping: {step_detail['input_mapping']}")
                    print(f"      Output ID: {step_detail['output_id']}")
                    print(f"      Agent Config Overrides: {step_detail['agent_config_overrides']}")
        
        if analysis_result.get("suggested_agents"):
            print(f"  Suggested Agents (capability-based): {analysis_result['suggested_agents']}")
    
    print("\n--- TaskAnalyzer refined demonstration with parallel plans completed. ---")
