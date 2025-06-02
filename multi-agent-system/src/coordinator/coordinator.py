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

        if len(selected_agents) == 1:
            primary_agent = selected_agents[0]
            self.logger.info(f"Dispatching query to primary selected agent: {primary_agent.get_name()}")

            # Prepare query_data for the single agent
            agent_query_data: Dict[str, Any] = {}
            processed_prompt = analysis_result.get("processed_query_for_agent", query)

            gemini_agent_class = self.agent_classes.get("gemini")
            is_gemini_instance = gemini_agent_class and isinstance(primary_agent, gemini_agent_class)

            if is_gemini_instance:
                agent_query_data["prompt_parts"] = [processed_prompt]
            else:
                agent_query_data["prompt"] = processed_prompt

            if analysis_result.get("system_prompt"):
                agent_query_data["system_prompt"] = analysis_result.get("system_prompt")

            if query_data_overrides:
                self.logger.info(f"Applying query_data_overrides for single agent: {query_data_overrides}")
                temp_overrides = query_data_overrides.copy()
                if is_gemini_instance:
                    if "prompt_parts" in temp_overrides:
                        agent_query_data["prompt_parts"] = temp_overrides.pop("prompt_parts")
                    if "prompt" in temp_overrides and "prompt_parts" in agent_query_data:
                        temp_overrides.pop("prompt", None)
                else:
                    if "prompt" in temp_overrides:
                        agent_query_data["prompt"] = temp_overrides.pop("prompt")
                    if "prompt_parts" in temp_overrides:
                        temp_overrides.pop("prompt_parts", None)
                agent_query_data.update(temp_overrides)
            else:
                self.logger.debug("No query_data_overrides provided for single agent.")

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

        elif len(selected_agents) > 1 and analysis_result.get("execution_plan"):
            # This branch handles rich, structured sequential execution if an execution_plan is provided
            # AND the routing engine returned a list of agents corresponding to that plan.
            # `selected_agents` here are the `planned_agent_instances` from RoutingEngine.

            execution_plan_details = analysis_result["execution_plan"] # List of step dicts
            self.logger.info(
                f"Starting rich sequential execution for {len(selected_agents)} agents based on execution_plan: "
                f"{[step.get('agent_name') for step in execution_plan_details]}"
            )

            execution_context: Dict[str, Any] = {} # To store outputs of steps by their output_id
            current_step_input_content: str = "" # Will be set based on input_mapping
            final_response_from_chain: Optional[Dict[str, Any]] = None

            for i, step_def in enumerate(execution_plan_details):
                if i >= len(selected_agents): # Should not happen if routing_engine validated plan
                    self.logger.error(f"Execution plan has more steps ({len(execution_plan_details)}) than selected agents ({len(selected_agents)}). Aborting.")
                    return {"status": "error", "message": "Execution plan and selected agents mismatch."}

                current_agent = selected_agents[i]
                agent_name_from_plan = step_def.get("agent_name")

                # Sanity check: agent from plan matches agent from selected_agents list
                if current_agent.get_name() != agent_name_from_plan:
                    self.logger.error(
                        f"Mismatch in execution: Step {i+1} expected agent '{agent_name_from_plan}' from plan, "
                        f"but router provided '{current_agent.get_name()}'. Aborting."
                    )
                    return {"status": "error", "message": "Execution plan and routed agent mismatch."}

                self.logger.info(
                    f"Executing step {i+1}/{len(execution_plan_details)} of plan: "
                    f"Agent: {current_agent.get_name()}, Task: {step_def.get('task_description', 'N/A')}"
                )

                # Determine input for the current step
                input_map = step_def.get("input_mapping", {})
                prompt_source = input_map.get("prompt_source")

                if prompt_source == "original_query":
                    current_step_input_content = str(analysis_result.get("processed_query_for_agent", query))
                elif prompt_source == "ref:previous_step.content":
                    if i == 0:
                        self.logger.error("First step in plan cannot reference 'previous_step.content'. Aborting.")
                        return {"status": "error", "message": "Invalid input_mapping for the first step of the plan."}

                    previous_step_output_id = execution_plan_details[i-1].get("output_id")
                    if not previous_step_output_id:
                        self.logger.error(f"Previous step (index {i-1}) is missing 'output_id'. Aborting.")
                        return {"status": "error", "message": "Previous step in plan missing 'output_id'."}

                    previous_response = execution_context.get(previous_step_output_id)
                    if not previous_response:
                        self.logger.error(f"Output from previous step ('{previous_step_output_id}') not found in execution_context. Aborting.")
                        return {"status": "error", "message": f"Missing output from previous step '{previous_step_output_id}'."}
                    if previous_response.get("status") != "success":
                        self.logger.error(f"Previous step ('{previous_step_output_id}') failed. Aborting current step. Details: {previous_response}")
                        return {"status": "error", "message": f"Previous step '{previous_step_output_id}' failed.", "details": previous_response}

                    content_from_prev_step = previous_response.get("content")
                    if content_from_prev_step is None:
                        self.logger.error(f"Previous step ('{previous_step_output_id}') succeeded but returned no 'content'. Aborting.")
                        return {"status": "error", "message": f"Previous step '{previous_step_output_id}' missing 'content'."}
                    current_step_input_content = str(content_from_prev_step)
                else:
                    self.logger.warning(
                        f"Unknown or missing 'prompt_source' in input_mapping for step {i+1} ('{step_def.get('task_description')}'). "
                        f"Defaulting to original processed query for this step."
                    )
                    current_step_input_content = str(analysis_result.get("processed_query_for_agent", query))

                # Prepare agent_query_data for the current agent in the plan
                agent_query_data: Dict[str, Any] = {}
                gemini_agent_class = self.agent_classes.get("gemini")
                is_gemini_instance = gemini_agent_class and isinstance(current_agent, gemini_agent_class)

                if is_gemini_instance:
                    agent_query_data["prompt_parts"] = [current_step_input_content]
                else:
                    agent_query_data["prompt"] = current_step_input_content

                # Apply query_data_overrides (e.g., temperature, max_tokens, global system_prompt)
                # These overrides apply to each step unless the step itself has more specific config from its template (not yet supported)
                if query_data_overrides:
                    temp_overrides = query_data_overrides.copy()
                    # Remove prompt/prompt_parts from general overrides as they are handled by chained content
                    temp_overrides.pop("prompt", None)
                    temp_overrides.pop("prompt_parts", None)
                    # System prompt from overrides applies if not already set by specific step logic (not yet supported)
                    # or if it's intended as a global override for the chain.
                    if "system_prompt" in temp_overrides:
                         agent_query_data["system_prompt"] = temp_overrides.pop("system_prompt")
                    agent_query_data.update(temp_overrides)

                # If no global system_prompt from overrides, and it's the first step,
                # use system_prompt from analysis_result if available.
                if "system_prompt" not in agent_query_data and i == 0 and analysis_result.get("system_prompt"):
                    agent_query_data["system_prompt"] = analysis_result.get("system_prompt")


                self.logger.debug(f"Query data for {current_agent.get_name()} (Plan Step {i+1}): {str(agent_query_data)[:200]}...")

                try:
                    response = current_agent.process_query(agent_query_data)
                    self.logger.info(f"Response from {current_agent.get_name()} (Plan Step {i+1}): {str(response)[:100]}...")
                except Exception as e:
                    self.logger.error(f"Error processing query with agent {current_agent.get_name()} in plan step {i+1}: {e}", exc_info=True)
                    return {"status": "error", "message": f"Error in plan step {i+1} with {current_agent.get_name()}: {str(e)}", "agent_name": current_agent.get_name()}

                output_id = step_def.get("output_id")
                if output_id:
                    execution_context[output_id] = response
                else:
                    self.logger.warning(f"Step {i+1} ({current_agent.get_name()}) missing 'output_id'. Its output cannot be referenced by subsequent steps.")

                final_response_from_chain = response # Store the latest response

                if response.get("status") != "success":
                    self.logger.warning(
                        f"Agent {current_agent.get_name()} in plan step {i+1} returned status '{response.get('status')}'. "
                        f"Message: {response.get('message')}. Halting plan execution."
                    )
                    return response

            return final_response_from_chain # Return the result of the last agent in the successful plan

        else: # Fallback to simple sequential if RoutingEngine returned multiple agents but there was no detailed plan
              # Or if only one agent was selected (len(selected_agents) == 1 was handled above)
              # This path should ideally not be hit if TaskAnalyzer always provides a plan or a single suggestion.
              # If RoutingEngine provides multiple agents without a plan, we revert to old simple sequential.
            self.logger.warning(
                f"Executing simple sequential mode for {len(selected_agents)} agents as no detailed execution_plan was found/processed."
            )
            current_input_content = str(analysis_result.get("processed_query_for_agent", query))
            legacy_final_response: Dict[str, Any] = {}
            for i, agent_instance in enumerate(selected_agents): # Should be only one if not a plan
                self.logger.info(f"Executing agent {i+1}/{len(selected_agents)} (legacy sequential): {agent_instance.get_name()}")
                agent_query_data = {"prompt": current_input_content} # Simplified for legacy path
                if query_data_overrides: # Basic override application
                    temp_overrides = query_data_overrides.copy()
                    temp_overrides.pop("prompt", None)
                    temp_overrides.pop("prompt_parts", None)
                    agent_query_data.update(temp_overrides)

                try:
                    response = agent_instance.process_query(agent_query_data)
                    legacy_final_response = response
                    if response.get("status") != "success" or response.get("content") is None:
                        self.logger.warning(f"Agent {agent_instance.get_name()} in legacy sequence failed or no content.")
                        return response
                    current_input_content = str(response.get("content"))
                except Exception as e:
                    self.logger.error(f"Error in legacy sequence with {agent_instance.get_name()}: {e}", exc_info=True)
                    return {"status": "error", "message": f"Error with {agent_instance.get_name()}: {str(e)}"}
            return legacy_final_response


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
