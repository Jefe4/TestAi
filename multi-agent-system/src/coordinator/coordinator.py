# src/coordinator/coordinator.py
"""
Main orchestrator for the multi-agent system.
Initializes agents, analyzes queries, routes tasks, and returns results.
"""

from typing import Dict, Any, List, Optional
import os # For GEMINI_API_KEY in __main__

try:
    from ..utils.api_manager import APIManager
    from ..utils.logger import get_logger
    from ..agents.base_agent import BaseAgent
    from .task_analyzer import TaskAnalyzer
    from .routing_engine import RoutingEngine
    from ..agents.deepseek_agent import DeepSeekAgent
    from ..agents.claude_agent import ClaudeAgent
    from ..agents.cursor_agent import CursorAgent # Assuming hypothetical API
    from ..agents.windsurf_agent import WindsurfAgent # Assuming hypothetical API
    from ..agents.gemini_agent import GeminiAgent
except ImportError:
    # Fallback for scenarios where the module might be run directly for testing
    # or if the PYTHONPATH is not set up correctly during development.
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils.api_manager import APIManager # type: ignore
    from src.utils.logger import get_logger # type: ignore
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.coordinator.task_analyzer import TaskAnalyzer # type: ignore
    from src.coordinator.routing_engine import RoutingEngine # type: ignore
    from src.agents.deepseek_agent import DeepSeekAgent # type: ignore
    from src.agents.claude_agent import ClaudeAgent # type: ignore
    from src.agents.cursor_agent import CursorAgent # type: ignore
    from src.agents.windsurf_agent import WindsurfAgent # type: ignore
    from src.agents.gemini_agent import GeminiAgent # type: ignore


class Coordinator:
    """
    Orchestrates task processing by analyzing queries, selecting appropriate
    agents, and dispatching tasks to them.
    """

    def __init__(self, agent_config_path: Optional[str] = None):
        """
        Initializes the Coordinator.

        Args:
            agent_config_path: Optional path to the main agent configuration file.
                               If None, APIManager will use its default path.
                               (Note: APIManager loads this, Coordinator uses APIManager's loaded configs)
        """
        self.logger = get_logger("Coordinator")

        # APIManager loads service configurations (which include agent configs) upon its own initialization.
        # If agent_config_path is provided, it implies that APIManager should use this path.
        # However, APIManager's __init__ already has a default path.
        # For clarity, if a specific path is given to Coordinator, we could pass it to APIManager,
        # but APIManager's current design loads from its DEFAULT_CONFIG_PATH or env vars.
        # Let's assume APIManager handles its config loading transparently.
        # If `agent_config_path` was meant for APIManager, it should be passed to APIManager's constructor.
        # For now, APIManager() will use its internal logic to find configs.
        self.api_manager = APIManager(config_path=agent_config_path) # Pass config_path to APIManager

        self.task_analyzer = TaskAnalyzer()
        self.routing_engine = RoutingEngine() # Can also take a config if needed in future
        self.agents: Dict[str, BaseAgent] = {}

        self.agent_classes: Dict[str, type[BaseAgent]] = {
            "deepseek": DeepSeekAgent,
            "claude": ClaudeAgent,
            "cursor": CursorAgent,
            "windsurf": WindsurfAgent,
            "gemini": GeminiAgent,
        }

        self._instantiate_agents()

        if self.agents:
            self.logger.info("Coordinator initialized successfully with agents: " + ", ".join(self.agents.keys()))
        else:
            self.logger.warning("Coordinator initialized, but no agents were instantiated. Check configurations.")


    def register_agent(self, agent_instance: BaseAgent):
        """
        Registers a new agent instance with the Coordinator.
        """
        agent_name = agent_instance.get_name()
        if agent_name in self.agents:
            self.logger.warning(f"Agent '{agent_name}' already registered. Overwriting with new instance.")
        self.agents[agent_name] = agent_instance
        self.logger.info(f"Agent '{agent_name}' registered successfully.")

    def _instantiate_agents(self):
        """
        Instantiates agents based on configurations loaded by the APIManager.
        APIManager's `service_configs` is expected to hold the configurations
        for services, which we interpret as agent configurations here.
        """
        self.logger.info("Attempting to instantiate agents based on loaded configurations...")

        # agent_configs are the service configurations loaded by APIManager
        # These configs are expected to contain API keys and any other specific settings for each service/agent.
        agent_configs = self.api_manager.service_configs

        if not agent_configs:
            self.logger.warning(
                "No agent configurations found in APIManager's service_configs. "
                "Cannot instantiate agents. Ensure 'agent_configs.yaml' is correctly "
                "populated and accessible, or environment variables are set for services."
            )
            return

        for agent_key, config_data in agent_configs.items():
            if not isinstance(config_data, dict):
                self.logger.warning(f"Configuration for agent key '{agent_key}' is not a dictionary. Skipping.")
                continue

            if agent_key in self.agent_classes:
                AgentClass = self.agent_classes[agent_key]

                # Agent name: use 'name' from config if present, else use the agent_key
                agent_name = config_data.get("name", agent_key)

                # Ensure API key is present in the config for the agent, otherwise skip (except for Gemini which checks env var too)
                # Most agents will need an API key from their config section.
                # GeminiAgent has its own logic to check self.config.get("api_key") or os.getenv("GEMINI_API_KEY")
                if agent_key != "gemini" and not config_data.get("api_key"):
                    self.logger.warning(
                        f"API key missing in configuration for agent '{agent_name}' (key: '{agent_key}'). Skipping instantiation. "
                        f"Ensure it's in agent_configs.yaml or corresponding environment variable for the service."
                    )
                    continue

                try:
                    # Pass the specific agent's config dict (config_data) and the shared api_manager
                    agent_instance = AgentClass(
                        agent_name=agent_name,
                        api_manager=self.api_manager,
                        config=config_data # Pass the specific config section for this agent
                    )
                    self.register_agent(agent_instance)
                except Exception as e:
                    self.logger.error(
                        f"Failed to instantiate agent '{agent_name}' (key: '{agent_key}') using class {AgentClass.__name__}: {e}",
                        exc_info=True
                    )
            else:
                self.logger.warning(f"No agent class mapping found for config key: '{agent_key}'. Skipping agent instantiation.")

        self.logger.info(f"Agent instantiation process completed. Current agents: {list(self.agents.keys())}")


    def process_query(self, query: str, query_data_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes a given query by analyzing it, selecting an agent,
        and dispatching the task to that agent.

        Args:
            query: The user's query string.
            query_data_overrides: Optional dictionary to pass additional structured data
                                  or overrides to the agent's process_query method.
                                  This can include 'system_prompt', 'max_tokens', etc.

        Returns:
            A dictionary containing the response from the agent or an error message.
        """
        self.logger.info(f"Coordinator received query: '{query[:100]}...'")

        if not self.agents:
            self.logger.error("No agents are registered or instantiated. Cannot process query.")
            return {"status": "error", "message": "No agents available in the system."}

        analysis_result = self.task_analyzer.analyze_query(query, self.agents)
        self.logger.debug(f"Task analysis result: {analysis_result}")

        selected_agents = self.routing_engine.select_agents(analysis_result, self.agents)

        if not selected_agents:
            self.logger.warning("Routing engine did not select any agent for the query.")
            return {"status": "error", "message": "No suitable agent found for the query."}

        # For now, use the first selected agent.
        # Future: Could involve strategies for multiple agents (e.g., consensus, chaining).
        primary_agent = selected_agents[0]
        self.logger.info(f"Dispatching query to primary selected agent: {primary_agent.get_name()}")

        # Prepare query_data for the agent
        agent_query_data: Dict[str, Any] = {}
        processed_prompt = analysis_result.get("processed_query_for_agent", query)

        # Check if the primary_agent is an instance of GeminiAgent using the stored class type
        gemini_agent_class = self.agent_classes.get("gemini")
        is_gemini_instance = gemini_agent_class and isinstance(primary_agent, gemini_agent_class)

        if is_gemini_instance:
            # Default to prompt_parts for Gemini
            agent_query_data["prompt_parts"] = [processed_prompt]
        else:
            # Default to prompt for other agents
            agent_query_data["prompt"] = processed_prompt

        # If TaskAnalyzer provides a specific system_prompt (not part of overrides yet)
        # This should be added before overrides to allow overrides to change it.
        # (Note: This was part of the old logic too, preserving its position relative to general prompt setting)
        if analysis_result.get("system_prompt"):
            agent_query_data["system_prompt"] = analysis_result.get("system_prompt")

        # Apply query_data_overrides
        if query_data_overrides:
            self.logger.info(f"Applying query_data_overrides: {query_data_overrides}")

            # Make a copy to safely pop items if needed for specific logic
            temp_overrides = query_data_overrides.copy()

            if is_gemini_instance:
                # If 'prompt_parts' is in overrides, it takes precedence for Gemini
                if "prompt_parts" in temp_overrides:
                    agent_query_data["prompt_parts"] = temp_overrides.pop("prompt_parts")
                # If 'prompt' is in overrides for Gemini, it might create ambiguity if not removed.
                # However, GeminiAgent itself should prioritize 'prompt_parts'.
                # If 'prompt' exists in temp_overrides, it will be added by update().
                # If 'prompt' was also in agent_query_data (e.g. from analysis_result for non-Gemini before override),
                # it would be overwritten by update().
                # For clarity, if 'prompt_parts' is set, we can remove 'prompt' from temp_overrides if it exists.
                if "prompt" in temp_overrides and "prompt_parts" in agent_query_data: # if prompt_parts was set either by default or override
                    temp_overrides.pop("prompt", None) # Remove 'prompt' from overrides if 'prompt_parts' is the primary way
            else: # Not Gemini
                # If 'prompt' is in overrides, it takes precedence for non-Gemini
                if "prompt" in temp_overrides:
                    # This was already set as default, so pop just ensures it's not processed again by update if logic changes.
                    # The agent_query_data["prompt"] would be updated by .update() anyway if key is present.
                    # To be explicit that override 'prompt' is used:
                    agent_query_data["prompt"] = temp_overrides.pop("prompt")
                # If 'prompt_parts' is in overrides for non-Gemini, it's ambiguous.
                # Non-Gemini agents are not expected to use 'prompt_parts'.
                # We can remove it from overrides to avoid confusion or let it pass.
                # Let's remove it to keep agent_query_data cleaner for non-Gemini.
                if "prompt_parts" in temp_overrides:
                    temp_overrides.pop("prompt_parts", None)


            agent_query_data.update(temp_overrides) # Apply remaining/all other overrides
        else:
            self.logger.debug("No query_data_overrides provided.")

        self.logger.debug(f"Prepared agent_query_data for {primary_agent.get_name()}: {agent_query_data}")

        try:
            response = primary_agent.process_query(agent_query_data)
            self.logger.info(f"Response from {primary_agent.get_name()} (first 100 chars): {str(response)[:100]}...")
            return response
        except Exception as e:
            self.logger.error(
                f"Error during query processing with agent {primary_agent.get_name()}: {e}",
                exc_info=True
            )
            return {"status": "error", "message": f"Failed to process query with {primary_agent.get_name()}: {str(e)}"}

if __name__ == '__main__':
    print("--- Coordinator Basic Test ---")

    # For this test to run meaningfully, it needs agent_configs.yaml or environment variables.
    # APIManager tries to load 'src/config/agent_configs.yaml' by default.
    # Let's create a dummy one if it doesn't exist for the __main__ block,
    # or mock APIManager's service_configs. Mocking is cleaner for a unit-like test.

    # Path for dummy config relative to this file's assumed location in src/coordinator
    dummy_config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
    dummy_config_file = os.path.join(dummy_config_dir, "agent_configs.yaml")

    # Ensure GEMINI_API_KEY is set for GeminiAgent instantiation if it's in dummy_config
    # For other agents, API keys should be in the dummy_config_file or their respective env vars.
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable not set. GeminiAgent might fail to initialize if configured.")
        # os.environ["GEMINI_API_KEY"] = "FAKE_GEMINI_KEY_FOR_TESTING_ONLY" # Not good practice to set here

    if not os.path.exists(dummy_config_file):
        print(f"Warning: Dummy config file {dummy_config_file} not found. Creating a minimal one for testing.")
        if not os.path.exists(dummy_config_dir):
            os.makedirs(dummy_config_dir)

        # Create a minimal dummy agent_configs.yaml for the test
        # Ensure keys match what agents expect (e.g., api_key, model)
        # And that the agent_key (e.g., 'deepseek', 'claude') matches self.agent_classes
        dummy_configs = {
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY_HERE"),
                "model": "deepseek-chat-test"
            },
            "claude": {
                "api_key": os.getenv("CLAUDE_API_KEY", "YOUR_CLAUDE_KEY_HERE"),
                "model": "claude-instant-test"
            },
            # GeminiAgent handles its own API key from env if not in config dict's "api_key"
            "gemini": {
                "model": "gemini-1.0-pro-test",
                # "api_key": os.getenv("GEMINI_API_KEY") # GeminiAgent will pick up from env
            }
            # Add other hypothetical agents if their classes exist and are imported
            # "cursor": {"api_key": "YOUR_CURSOR_KEY", "model": "cursor-alpha"},
            # "windsurf": {"api_key": "YOUR_WINDSURF_KEY", "focus": "general"},
        }
        import yaml
        with open(dummy_config_file, 'w') as f:
            yaml.dump(dummy_configs, f)
        print(f"Created dummy config: {dummy_config_file} with keys: {list(dummy_configs.keys())}")
        print("Please ensure API keys are set as environment variables (e.g., DEEPSEEK_API_KEY, CLAUDE_API_KEY, GEMINI_API_KEY) or in the YAML for agents to initialize.")

    print(f"Initializing Coordinator (will use config from: {dummy_config_file if os.path.exists(dummy_config_file) else 'APIManager defaults/env vars'})...")

    # If specific API keys are not set as ENV VARS or in the dummy file,
    # the agents requiring them might fail to initialize or their process_query calls might fail.
    # The GeminiAgent will raise ValueError if its key is not found. Others might try to use placeholder keys.
    try:
        coordinator = Coordinator(agent_config_path=dummy_config_file) # Tell APIManager to use this specific file

        if not coordinator.agents:
            print("Coordinator initialized, but no agents were loaded. Check config and API keys.")
        else:
            print(f"Coordinator initialized with agents: {list(coordinator.agents.keys())}")

            # Test query 1
            query1 = "What is the capital of France? Explain in one sentence."
            print(f"\nProcessing query 1: '{query1}'")
            response1 = coordinator.process_query(query1)
            print(f"Response 1: {response1}")

            # Test query 2 (code related)
            query2 = "Write a Python function to calculate factorial."
            print(f"\nProcessing query 2: '{query2}'")
            response2 = coordinator.process_query(query2)
            print(f"Response 2: {response2}")

            # Test query 3 (Gemini specific, if Gemini agent is loaded)
            if "gemini" in coordinator.agents:
                 query3 = "Describe this image." # Needs actual image data for real use
                 print(f"\nProcessing query 3 (for Gemini): '{query3}'")
                 # For a real multimodal query, prompt_parts would include image data.
                 # Here, we send it as text, which Gemini can still handle.
                 response3 = coordinator.process_query(query3, query_data_overrides={"prompt_parts": ["Describe a cat."]})
                 print(f"Response 3: {response3}")
            else:
                print("\nGemini agent not loaded, skipping Gemini-specific query test.")

    except ValueError as ve:
        print(f"A ValueError occurred during Coordinator/Agent initialization: {ve}")
        print("This often means an API key is missing for an agent (like Gemini).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", exc_info=True)

    print("\n--- Coordinator Basic Test Finished ---")
    # Consider removing the dummy config file after test if it was created by this script
    # if os.path.exists(dummy_config_file) and "deepseek-chat-test" in str(dummy_configs.get("deepseek",{})):
    #     # os.remove(dummy_config_file)
    #     # print(f"Removed dummy config file: {dummy_config_file}")
    #     pass # Decided not to auto-delete for easier re-testing / inspection.
    print(f"Note: If a dummy config was created at {dummy_config_file}, it will persist.")
