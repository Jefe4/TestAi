# src/utils/context_manager.py
"""
Provides the AdvancedContextManager for tracking and managing execution context (trace)
within the multi-agent system. This can include logging events, retrieving trace history,
and potentially compressing context for use by agents.
"""

from typing import List, Dict, Any, Optional
import datetime

# Attempt to import APIManager for context compression and logger.
# Fallback definitions are provided if run in an environment where these imports might fail.
try:
    from .logger import get_logger
    from .api_manager import APIManager
except ImportError:
    import logging
    # Basic placeholder for APIManager if the real one isn't available.
    # This allows the module to be imported but compression will fail gracefully.
    class APIManager: # type: ignore
        def __init__(self, *args, **kwargs):
            self.service_configs: Dict[str, Any] = {} # Ensure service_configs exists for dummy
        async def make_request(self, *args, **kwargs) -> Dict[str, Any]:
            # Dummy make_request for fallback; real compression needs a functional APIManager.
            return {"status": "error", "message": "Fallback APIManager: make_request not implemented."}

    def get_logger(name): # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class AdvancedContextManager:
    """
    Manages an execution trace and provides utilities for context manipulation.

    The trace is a list of timestamped events, where each event is a dictionary
    containing an `event_type`, `data`, and any other relevant keyword arguments.
    This manager can also compress the trace history using an LLM via APIManager.
    """
    def __init__(self, api_manager: Optional[APIManager] = None):
        """
        Initializes the AdvancedContextManager.

        Args:
            api_manager: An optional instance of APIManager. If provided,
                         it enables context compression capabilities.
        """
        self.logger = get_logger("AdvancedContextManager")
        self._trace: List[Dict[str, Any]] = [] # Internal list to store trace events
        self.api_manager = api_manager # Store the APIManager instance for compression

        if not self.api_manager:
            self.logger.warning(
                "APIManager not provided to AdvancedContextManager. "
                "Context compression feature will be disabled."
            )
        self.logger.info("AdvancedContextManager initialized.")

    def add_trace_event(self, event_type: str, data: Dict[str, Any], **kwargs: Any) -> None:
        """
        Adds a new event to the execution trace.

        Each event is timestamped (UTC) and includes the event type,
        associated data, and any additional keyword arguments passed.

        Args:
            event_type: A string describing the type of event (e.g., "agent_call_start").
            data: A dictionary containing data relevant to the event.
            **kwargs: Arbitrary additional key-value pairs to include in the event.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event: Dict[str, Any] = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data, # Main data payload for the event
            **kwargs     # Include any other keyword arguments in the event
        }
        self._trace.append(event)
        self.logger.debug(f"Added trace event: Type='{event_type}', Data='{str(data)[:100]}...'")

    def get_full_trace(self) -> List[Dict[str, Any]]:
        """
        Retrieves a copy of the entire execution trace.

        Returns:
            A list of all event dictionaries recorded in the trace.
            A copy is returned to prevent external modification of the internal trace.
        """
        return self._trace.copy()

    def get_recent_context(self, n_events: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves a copy of the last 'n_events' from the execution trace.

        Args:
            n_events: The number of recent events to retrieve. Defaults to 5.

        Returns:
            A list of the most recent 'n_events' dictionaries. Returns an empty
            list if `n_events` is not positive or if the trace is shorter.
        """
        if n_events <= 0:
            return []
        return self._trace[-n_events:].copy()

    def clear_trace(self) -> None:
        """Clears all events from the current execution trace."""
        self._trace = []
        self.logger.info("Execution trace has been cleared.")

    async def compress_context(self,
                               history: Optional[List[Dict[str, Any]]] = None,
                               model_name: str = "claude-3-haiku-20240307") -> str:
        """
        Compresses a given history of trace events (or the current trace if None)
        into a concise summary using a Language Model (LLM) via the APIManager.

        Args:
            history: An optional list of event dictionaries to compress. If None,
                     the current internal trace (`self._trace`) is used.
            model_name: The name of the LLM model to use for compression
                        (defaulting to a Claude Haiku model).

        Returns:
            A string containing the compressed summary of the history, or an
            error message if compression fails or is unavailable.
        """
        if not self.api_manager:
            self.logger.error("Cannot compress context: APIManager instance is not available.")
            return "Error: Context compression unavailable (APIManager not provided to ContextManager)."

        target_history = history if history is not None else self._trace
        if not target_history:
            self.logger.info("No history provided or available in trace to compress.")
            return "No history to compress."

        # Format the history into a single string for the LLM prompt.
        # Each event is represented as "- event_type: data_summary..."
        # Data is truncated to avoid excessively long individual event strings.
        history_string = "\n".join(
            [f"- {event.get('event_type', 'unknown_event')}: {str(event.get('data', {}))[:200]}"
             for event in target_history]
        )

        # Truncate the overall history string if it's too long for the LLM prompt.
        max_history_len = 10000  # Define a safe maximum length for the history part of the prompt
        if len(history_string) > max_history_len:
            history_string = history_string[-max_history_len:] # Keep the most recent part
            self.logger.warning(f"History string for compression was truncated to the last {max_history_len} characters.")

        # Construct the prompt for the LLM.
        compression_prompt = f"""
Please compress the following execution trace into a concise summary of key decisions, critical context, and essential outcomes.
Preserve all information crucial for future decisions by an AI agent. Focus on the flow of actions and results.
Avoid conversational fluff. Aim for a structured summary if possible, perhaps using bullet points for key items.

Execution Trace:
{history_string}

Concise Summary:
"""

        self.logger.info(f"Attempting context compression using model '{model_name}' for {len(target_history)} events.")

        try:
            # Check if the required service (e.g., "claude") is configured in APIManager
            if not hasattr(self.api_manager, 'service_configs') or "claude" not in self.api_manager.service_configs:
                 self.logger.error(f"Service 'claude' (for model '{model_name}') not configured in APIManager. Cannot compress context.")
                 return f"Error: Service 'claude' for compression (model {model_name}) not configured."

            # Make the API call for compression
            response = await self.api_manager.make_request(
                service_name="claude", # Hardcoded to 'claude' for now, assuming it hosts the compression model
                endpoint="messages",   # Standard endpoint for Claude messages API
                method="POST",
                data={
                    "model": model_name, # Specify the compression model
                    "messages": [{"role": "user", "content": compression_prompt}],
                    "max_tokens": 1000,  # Max tokens for the generated summary
                    "temperature": 0.2,  # Lower temperature for more factual, less creative summary
                }
            )

            # Process the response
            if response.get("status") == "success" and response.get("content"):
                compressed_text = ""
                # Standard Claude API Messages response structure often has content in a list of dicts
                if isinstance(response.get("content"), list) and len(response.get("content")) > 0:
                    content_block = response.get("content")[0]
                    if isinstance(content_block, dict) and "text" in content_block:
                         compressed_text = content_block.get("text","")
                elif isinstance(response.get("content"), str): # Fallback if content is already a string
                    compressed_text = response.get("content")

                if not compressed_text:
                    self.logger.error(f"Compression call succeeded but no text found in response. Response (first 300 chars): {str(response)[:300]}")
                    return f"Error: Compression failed (empty or malformed response content). Original history length: {len(target_history)} events."

                self.logger.info("Context compression successful.")
                return compressed_text.strip() # Return stripped text
            else:
                # Handle API call failure or error status in response
                error_msg = response.get("message", str(response.get("details", "Unknown error during compression API call.")))
                self.logger.error(f"Context compression API call failed: {error_msg}")
                return f"Error: Context compression API call failed. Details: {error_msg}. Original history length: {len(target_history)} events."

        except Exception as e: # Catch any other exceptions during the process
            self.logger.error(f"Exception during context compression: {e}", exc_info=True)
            return f"Error: Exception during context compression - {str(e)}. Original history length: {len(target_history)} events."

if __name__ == '__main__':
    import asyncio

    # Dummy APIManager for testing compression (won't make real calls unless configured)
    # Need to ensure the fallback APIManager is correctly defined if the real one isn't imported
    if 'APIManager' not in globals() or not hasattr(APIManager, 'make_request'): # If using fallback APIManager
        class DummyAPIManager: # type: ignore
            def __init__(self, *args, **kwargs):
                self.logger = get_logger("DummyAPIManager_ContextTest")
                self.service_configs: Dict[str, Any] = {} # Ensure service_configs exists
            async def make_request(self, service_name: str, endpoint: str, method: str = "POST", data: Optional[Dict[str, Any]] = None, **kwargs):
                if service_name == "claude" and endpoint == "messages":
                    prompt_content = data.get("messages", [{}])[0].get("content", "")
                    self.logger.info(f"DummyAPIManager received compression request. Prompt length: {len(prompt_content)}")
                    return {
                        "status": "success",
                        "content": [{"type":"text", "text":f"Compressed summary of: {prompt_content[:100]}..."}],
                        "usage": {"input_tokens": 50, "output_tokens": 10}
                    }
                return {"status": "error", "message": "DummyAPIManager: Service/endpoint not mocked for this test."}
    else: # If real APIManager was imported
        class DummyAPIManager(APIManager): # type: ignore
            async def make_request(self, service_name: str, endpoint: str, method: str = "POST", data: Optional[Dict[str, Any]] = None, **kwargs):
                if service_name == "claude" and endpoint == "messages":
                    prompt_content = data.get("messages", [{}])[0].get("content", "")
                    self.logger.info(f"DummyAPIManager (subclass) received compression request. Prompt length: {len(prompt_content)}")
                    return {
                        "status": "success",
                        "content": [{"type":"text", "text":f"Compressed summary of: {prompt_content[:100]}..."}],
                        "usage": {"input_tokens": 50, "output_tokens": 10}
                    }
                # Call super for other cases or return error
                # return await super().make_request(service_name, endpoint, method, data, **kwargs)
                return {"status": "error", "message": "DummyAPIManager (subclass): Service/endpoint not mocked."}


    async def main_context_test(): # Wrap in async main
        api_manager_instance = DummyAPIManager()

        if "claude" not in api_manager_instance.service_configs:
            api_manager_instance.service_configs["claude"] = {"api_key": "dummy_key", "base_url":"dummy_url"}

        context_manager = AdvancedContextManager(api_manager=api_manager_instance)

        context_manager.add_trace_event("initial_query", {"query_text": "What is the weather in London?"})
        context_manager.add_trace_event("task_analysis", {"type": "weather_query", "location": "London"})
        context_manager.add_trace_event(
            "agent_call_start",
            {"agent_name": "WeatherAgent", "input": "London"},
            step_id="weather_step_1"
        )
        context_manager.add_trace_event(
            "agent_call_end",
            {"agent_name": "WeatherAgent", "output": {"temperature": "15C", "condition": "Cloudy"}},
            step_id="weather_step_1",
            status="success"
        )
        context_manager.add_trace_event("final_response", {"response_text": "The weather in London is 15C and Cloudy."})

        print("--- Full Trace ---")
        full_trace = context_manager.get_full_trace()
        for event in full_trace:
            print(event)

        print("\n--- Recent Context (Last 3 Events) ---")
        recent_context = context_manager.get_recent_context(n_events=3)
        for event in recent_context:
            print(event)

        print("\n--- Compressing Current Context ---")
        compressed_summary = await context_manager.compress_context()
        print(f"Compressed Summary: {compressed_summary}")

        print("\n--- Compressing Specific History (last 2 events) ---")
        specific_history = context_manager.get_recent_context(n_events=2)
        compressed_specific = await context_manager.compress_context(history=specific_history)
        print(f"Compressed Specific Summary: {compressed_specific}")

        context_manager.clear_trace()
        print("\n--- Trace after clearing ---")
        print(context_manager.get_full_trace())

    asyncio.run(main_context_test())
