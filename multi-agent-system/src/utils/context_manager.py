from typing import List, Dict, Any, Optional # Added Optional
import datetime

# Assuming logger might be needed, similar to other utils
try:
    from .logger import get_logger
    from .api_manager import APIManager # Added
except ImportError: # Fallback for direct execution or if logger isn't in the same dir as expected
    import logging
    # Define APIManager and get_logger as placeholders for fallback if needed
    class APIManager: # type: ignore
        def __init__(self, *args, **kwargs): pass
        async def make_request(self, *args, **kwargs): return {}

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
    def __init__(self, api_manager: Optional[APIManager] = None): # Modified signature
        self.logger = get_logger("AdvancedContextManager")
        self._trace: List[Dict[str, Any]] = []
        self.api_manager = api_manager # Stored APIManager
        if not self.api_manager:
            self.logger.warning("APIManager not provided to AdvancedContextManager. Context compression will be disabled.")
        self.logger.info("AdvancedContextManager initialized.")

    def add_trace_event(self, event_type: str, data: Dict[str, Any], **kwargs: Any):
        """Adds an event to the execution trace."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": data,
            **kwargs # Allow arbitrary extra fields for specific events
        }
        self._trace.append(event)
        self.logger.debug(f"Added trace event: {event_type} - {str(data)[:100]}...")

    def get_full_trace(self) -> List[Dict[str, Any]]:
        """Returns the entire execution trace."""
        return self._trace.copy() # Return a copy to prevent external modification

    def get_recent_context(self, n_events: int = 5) -> List[Dict[str, Any]]:
        """Returns the last n_events from the trace."""
        return self._trace[-n_events:].copy() if n_events > 0 else []

    def clear_trace(self):
        """Clears the current trace."""
        self._trace = []
        self.logger.info("Execution trace cleared.")

    async def compress_context(self, history: Optional[List[Dict[str, Any]]] = None, model_name: str = "claude-3-haiku-20240307") -> str:
        """
        Compresses the given history (or the current trace if None) using an LLM.
        Returns the compressed text or an error message if compression fails.
        """
        if not self.api_manager:
            self.logger.error("Cannot compress context: APIManager not available.")
            return "Error: Context compression unavailable (APIManager missing)."

        target_history = history if history is not None else self._trace
        if not target_history:
            return "No history to compress."

        # Simple formatting of history to string. Could be more sophisticated.
        history_string = "\n".join([f"- {event['event_type']}: {str(event['data'])[:200]}" for event in target_history])

        # Limit history string length to avoid overly long prompts (e.g. 10k chars for safety)
        max_history_len = 10000
        if len(history_string) > max_history_len:
            history_string = history_string[-max_history_len:] # Take the most recent part
            self.logger.warning(f"History string truncated to last {max_history_len} chars for compression prompt.")

        compression_prompt = f"""
Please compress the following execution trace into a concise summary of key decisions, critical context, and essential outcomes.
Preserve all information crucial for future decisions by an AI agent. Focus on the flow of actions and results.
Avoid conversational fluff. Aim for a structured summary if possible.

Execution Trace:
{history_string}

Concise Summary:
"""

        self.logger.info(f"Attempting context compression with model {model_name} for {len(target_history)} events.")

        try:
            # Ensure APIManager is available and the service is configured
            # Assuming service_configs is accessible on APIManager instance
            if not hasattr(self.api_manager, 'service_configs') or "claude" not in self.api_manager.service_configs:
                 self.logger.error("Claude service not configured in APIManager. Cannot compress context.")
                 return "Error: Claude service not configured for compression."

            response = await self.api_manager.make_request(
                service_name="claude", # Assuming 'claude' is configured in APIManager
                endpoint="messages", # Standard Claude messages endpoint
                method="POST",
                data={
                    "model": model_name,
                    "messages": [{"role": "user", "content": compression_prompt}],
                    "max_tokens": 1000, # Max tokens for the summary
                    "temperature": 0.2, # Lower temperature for more factual summary
                }
            )

            if response.get("status") == "success" and response.get("content"):
                compressed_text = ""
                # Standard Claude API response for messages
                if isinstance(response.get("content"), list) and len(response.get("content")) > 0:
                    if isinstance(response.get("content")[0], dict) and "text" in response.get("content")[0]:
                         compressed_text = response.get("content")[0].get("text","")
                elif isinstance(response.get("content"), str): # Fallback if content is just a string (less likely for Claude messages)
                    compressed_text = response.get("content")

                if not compressed_text:
                    self.logger.error(f"Compression call succeeded but no text found in response. Response: {str(response)[:300]}")
                    return f"Error: Compression failed (empty response content). Original history length: {len(target_history)} events."

                self.logger.info("Context compression successful.")
                return compressed_text
            else:
                error_msg = response.get("message", str(response.get("details", "Unknown error during compression.")))
                self.logger.error(f"Context compression failed: {error_msg}")
                return f"Error: Context compression failed. Details: {error_msg}. Original history length: {len(target_history)} events."
        except Exception as e:
            self.logger.error(f"Exception during context compression: {e}", exc_info=True)
            return f"Error: Exception during context compression - {str(e)}. Original history length: {len(target_history)} events."

if __name__ == '__main__':
    import asyncio # Add this import

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
