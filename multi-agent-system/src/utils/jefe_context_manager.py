# multi-agent-system/src/utils/jefe_context_manager.py
from typing import List, Dict, Any, Optional
import asyncio # For asyncio.get_event_loop().time() and potential async operations
import datetime # For fallback timestamp if event loop time isn't ideal

from .jefe_datatypes import JefeContext # Assuming jefe_datatypes.py is in the same directory
# Try to import APIManager, provide a basic fallback if not found (e.g., for standalone testing)
try:
    from .api_manager import APIManager
except ImportError:
    import logging as logger_fallback # Use a different name to avoid conflict
    class APIManager: # type: ignore
        """Fallback APIManager if the real one cannot be imported."""
        def __init__(self, *args, **kwargs):
            self.logger = logger_fallback.getLogger("FallbackAPIManager")
            self.logger.warning("Using FallbackAPIManager. Real API calls will not work.")
            self.service_configs: Dict[str, Any] = {} # For dummy testing
        async def call_llm_service(self, *args, **kwargs) -> Dict[str, Any]:
            self.logger.warning("FallbackAPIManager: call_llm_service called, but will return error.")
            return {"status": "error", "message": "Using FallbackAPIManager, real call_llm_service not available."}


# Assuming a logger utility exists
try:
    from .logger import get_logger
except ImportError:
    import logging
    def get_logger(name): # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers: # Basic setup if no handlers configured
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

class JefeContextManager:
    """
    Manages context, conversation history, compressed memory, and project-specific
    understanding for the JefeAgent. Implements principles for context engineering.
    """
    def __init__(self, api_manager: APIManager, compression_model: str = "claude-3-haiku-20240307"):
        self.logger = get_logger("JefeContextManager")
        self.api_manager = api_manager
        self.compression_model = compression_model

        self.conversation_history: List[Dict[str, Any]] = [] # Stores {'context': JefeContext, 'response': Dict, 'timestamp': float}
        self.compressed_memory: Dict[str, str] = {} # Stores summaries, e.g., by timestamp or topic
        self.active_projects: Dict[str, Dict[str, Any]] = {} # Stores project-specific distilled context

        self.max_history_len_before_compression = 50 # Number of interactions
        self.recent_interactions_to_keep = 20 # Number of recent interactions to keep uncompressed

        self.logger.info(f"JefeContextManager initialized. Max history: {self.max_history_len_before_compression}, Keep recent: {self.recent_interactions_to_keep}, Compression Model: {self.compression_model}")

    def add_interaction(self, context: JefeContext, response: Dict[str, Any]):
        """Adds a new interaction (context + response) to the history."""
        try:
            # Use event loop time if available for higher precision and monotonicity within a session
            timestamp = asyncio.get_event_loop().time()
        except RuntimeError: # Fallback if no event loop is running (e.g. some test scenarios, or synchronous parts)
            timestamp = datetime.datetime.now(datetime.timezone.utc).timestamp()

        interaction = {
            "context": context, # Storing the JefeContext object itself
            "response": response,
            "timestamp": timestamp
        }
        self.conversation_history.append(interaction)
        self.logger.debug(f"Added interaction. History length: {len(self.conversation_history)}")

        # Check if history exceeds threshold and schedule compression
        if len(self.conversation_history) > self.max_history_len_before_compression:
            # Schedule _compress_old_context to run in the background without blocking add_interaction
            asyncio.create_task(self._compress_old_context())

        self._update_project_context(context) # Update project-specific understanding


    async def _compress_old_context(self):
        """Compresses older parts of the conversation history if APIManager is available."""
        if not self.api_manager or not isinstance(self.api_manager, APIManager): # Check if it's the real APIManager
            self.logger.warning("APIManager not available or is a fallback instance, skipping context compression.")
            return

        if len(self.conversation_history) <= self.recent_interactions_to_keep:
            self.logger.debug("Not enough history to trigger compression beyond recent interactions.")
            return

        num_to_compress = len(self.conversation_history) - self.recent_interactions_to_keep
        if num_to_compress <= 0: # Should be caught by above, but defensive check
            return

        to_compress_interactions = self.conversation_history[:num_to_compress]
        # Keep the most recent interactions in the main history
        self.conversation_history = self.conversation_history[num_to_compress:]

        self.logger.info(f"Attempting to compress {len(to_compress_interactions)} old interactions...")

        # Format history for the LLM. Using summaries from JefeContext.
        formatted_history_for_llm: List[str] = []
        for interaction in to_compress_interactions:
            context_obj = interaction.get('context')
            response_obj = interaction.get('response', {})

            ctx_summary = "No context"
            if isinstance(context_obj, JefeContext):
                ctx_summary = context_obj.summarize(max_screen_len=200, max_audio_len=100) # Use summarize method
            elif isinstance(context_obj, dict): # Fallback if context is a dict
                ctx_summary = str(context_obj)[:200]

            resp_content = response_obj.get('content', response_obj.get('response', 'No response content.'))
            resp_summary = str(resp_content)[:200]

            formatted_history_for_llm.append(f"Context Snapshot:\n{ctx_summary}\n\nJefe Response:\n{resp_summary}")

        history_str = "\n\n===\nNext Interaction\n===\n\n".join(formatted_history_for_llm)
        if not history_str:
            self.logger.info("No content to compress after formatting interactions.")
            return

        try:
            # Construct messages for call_llm_service
            # The prompt asks the LLM to summarize the provided interaction history.
            compression_task_prompt = (
                "Concisely summarize the key information, decisions, user queries, AI responses, and outcomes "
                "from the following interaction history. Focus on details that would be important for an AI assistant "
                "to remember for maintaining context in an ongoing session. Discard conversational fluff or "
                "highly verbose data. Present as a dense summary or key bullet points highlighting critical information."
                f"\n\nInteraction History:\n{history_str}"
            )
            messages = [{"role": "user", "content": compression_task_prompt}]

            compression_result = await self.api_manager.call_llm_service(
                service_name="claude", # Default or make this configurable via self.compression_service
                model_name=self.compression_model,
                messages=messages,
                max_tokens=1500, # Adjust token limit for summary based on expected density
                temperature=0.1  # Low temperature for factual, dense summarization
            )

            if compression_result.get("status") == "success" and compression_result.get("content"):
                compressed_text = compression_result["content"]
                # Use timestamp of the last compressed item as part of the key for uniqueness
                summary_key = f"summary_ended_at_{to_compress_interactions[-1]['timestamp']:.0f}"
                self.compressed_memory[summary_key] = compressed_text
                self.logger.info(f"Context compression successful. Summary '{summary_key}' has {len(compressed_text)} chars.")
            else:
                self.logger.error(f"LLM call for context compression failed or returned empty content: {compression_result.get('message', 'No content')}")
        except Exception as e:
            self.logger.error(f"Exception during LLM context compression: {e}", exc_info=True)
            # Strategy for handling failed compression:
            # Option 1: Add back to_compress_interactions to self.conversation_history (might lead to repeated attempts).
            # Option 2: Store them in a temporary "failed_compression" list for later retry.
            # Option 3: Log and discard (current implicit behavior if not re-added).
            # For now, they are removed from active history and only a log is made. A more robust system might retry.

    def _update_project_context(self, context: JefeContext):
        """
        Placeholder for logic to understand and store project-specific context.
        This could involve identifying project root, frameworks, key files, common
        terms, or user goals related to a specific project.

        Args:
            context: The current JefeContext object.
        """
        # Example: Use project_type or a hash of a key file path as a project_key
        project_key = context.project_type or "general_coding"
        if context.current_ide:
            project_key = f"{context.current_ide}_{project_key}"

        if project_key not in self.active_projects:
            self.active_projects[project_key] = {
                "languages": set(),
                "common_terms": set(), # Could be populated by analyzing screen/audio for keywords
                "error_patterns": set(),
                "key_files": set()
            }

        if context.programming_language:
            self.active_projects[project_key]["languages"].add(context.programming_language)
        if context.error_messages:
            for err in context.error_messages: # Add first line of error as pattern
                 self.active_projects[project_key]["error_patterns"].add(err.split('\n')[0])

        # This is a very basic example. More sophisticated analysis would be needed here.
        self.logger.debug(f"Updated active project context for key '{project_key}'. Current state: {self.active_projects[project_key]}")


    def get_full_conversation_history(self) -> List[Dict[str, Any]]:
        """Returns a copy of the current (uncompressed) conversation history."""
        return self.conversation_history.copy()

    def get_recent_interactions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Returns a copy of the last 'n' interactions from the uncompressed history."""
        return self.conversation_history[-n:].copy() if n > 0 else []

    def get_compressed_memory_summary(self) -> str:
        """
        Returns a concatenation of all compressed memory segments.
        Order might matter, so consider sorting keys if timestamps are not strictly sequential.
        """
        # Sort by the timestamp in the key for chronological order of summaries
        sorted_summary_keys = sorted(self.compressed_memory.keys())
        return "\n\n---\n[Compressed Segment]\n---\n\n".join(
            self.compressed_memory[key] for key in sorted_summary_keys
        )

    def get_relevant_context_summary(self, current_query: str, project_key: Optional[str] = None) -> str:
        """
        Constructs a relevant context summary for a new query.
        Includes recent interactions and relevant compressed memory/project context.
        (This is a more advanced method for future implementation)
        """
        # 1. Get recent interactions
        recent_str = "\n".join([f"Context: {i['context'].summarize(100,50)}\nJefe: {str(i['response'].get('content',''))[:100]}"
                                for i in self.get_recent_interactions(3)])

        # 2. Get compressed memory
        compressed_mem = self.get_compressed_memory_summary()

        # 3. Get project specific context (simplified)
        project_info_str = "No specific project context."
        if project_key and project_key in self.active_projects:
            proj_data = self.active_projects[project_key]
            project_info_str = f"Current Project ({project_key}): Langs - {', '.join(list(proj_data['languages'])) if proj_data['languages'] else 'N/A'}. Recent Errors - {', '.join(list(proj_data['error_patterns'])[-2:]) if proj_data['error_patterns'] else 'N/A'}."

        # 4. Combine them (this is a simple combination, could be more intelligent)
        return f"Current Query: {current_query}\n\nRecent Interactions:\n{recent_str}\n\nProject Context:\n{project_info_str}\n\nLong-term Compressed Memory Summary:\n{compressed_mem}"


if __name__ == '__main__':
    # Ensure the fallback APIManager is used if the real one isn't available/configured for testing.
    # This setup is primarily for demonstrating JefeContextManager's own logic.
    if 'APIManager' not in globals() or APIManager.__module__ == __name__: # Check if using fallback
        MockAPIManager = APIManager # Use the fallback defined in this file
    else: # If real APIManager was imported, create a mock wrapper or use as is if configured
        from .api_manager import APIManager as RealAPIManager
        class MockAPIManager(RealAPIManager):
            async def call_llm_service(self, service_name: str, model_name: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float = 0.3, **kwargs: Any) -> Dict[str, Any]:
                self.logger.info(f"MockAPIManager (subclass of real): call_llm_service for {service_name}, model {model_name}")
                if messages and messages[0].get("content"):
                    summary_text = f"Mock compressed summary of: {messages[0]['content'][:150]}..."
                    # Simulate Claude's successful response structure for content extraction
                    return {"status": "success", "content": summary_text }
                return {"status": "error", "content": "Mock error: No messages provided to mock."}


    async def main_jefe_context_test():
        print("--- JefeContextManager Test ---")

        # Instantiate the (potentially mock) APIManager
        # If using the real APIManager, ensure 'claude' service is configured in agent_configs.yaml for compression.
        mock_api_manager = MockAPIManager()
        # Ensure claude service is "configured" for the mock/dummy APIManager if it checks service_configs
        if not hasattr(mock_api_manager, 'service_configs'):
            mock_api_manager.service_configs = {} # type: ignore
        if "claude" not in mock_api_manager.service_configs:
             mock_api_manager.service_configs["claude"] = {"api_key":"dummy_claude_key_for_ctx_mgr_test", "base_url":"http://localhost/dummy_claude"}

        manager = JefeContextManager(api_manager=mock_api_manager, compression_model="test-compression-model")
        manager.max_history_len_before_compression = 3 # Trigger compression quickly for test
        manager.recent_interactions_to_keep = 1      # Keep only 1 recent interaction after compression

        # Simulate some interactions
        print("\nAdding interactions...")
        for i in range(4): # Add 4 interactions, will trigger compression
            ctx = JefeContext(
                screen_content=f"Screen content for interaction {i+1}. User is editing function 'calculate_total_{i}'. Error on line {i*5}.",
                audio_transcript=f"User: How do I fix this error in calculate_total_{i}? It says 'undefined variable'.",
                current_ide="VSCode",
                programming_language="Python",
                project_type="DataProcessingScript",
                error_messages=[f"Error_code_XYZ{i}: undefined variable 'total_value' on line {i*5}"]
            )
            response = {"status":"success", "content": f"Jefe response for issue {i+1}: Suggested checking variable scope.", "task_type": JefeTaskType.DEBUGGING_ASSISTANCE.value}
            manager.add_interaction(ctx, response)
            print(f"Added interaction {i+1}. History length: {len(manager.get_full_conversation_history())}")
            # Brief pause to allow any created compression tasks to potentially start or complete
            # In a real app, these tasks run in the background. For testing, we want to see results.
            await asyncio.sleep(0.1)


        print(f"\nTotal History length after additions: {len(manager.get_full_conversation_history())}")

        # It's tricky to deterministically test asyncio.create_task execution timing within a simple script.
        # To ensure compression runs for the test, we can call it directly.
        # Note: In a live application, create_task is usually fine.
        if len(manager.conversation_history) > manager.recent_interactions_to_keep:
             print("\nManually triggering compression for test verification...")
             await manager._compress_old_context()

        print(f"\nHistory length after explicit compression call (if any): {len(manager.get_full_conversation_history())}")

        recent_interactions = manager.get_recent_interactions(n=manager.recent_interactions_to_keep) # Get what should remain
        print(f"\nRecent {len(recent_interactions)} interactions kept:")
        for idx, interaction in enumerate(recent_interactions):
            print(f"  Interaction -{len(recent_interactions)-idx}:")
            print(f"    Context Summary: {interaction['context'].summarize(100, 50)}")
            print(f"    Response: {str(interaction['response'].get('content'))[:100]}...")

        compressed_summary_text = manager.get_compressed_memory_summary()
        print(f"\nFull Compressed Memory Summary ({len(manager.compressed_memory)} segments):")
        print(compressed_summary_text if compressed_summary_text else "No compressed memory yet.")

        print(f"\nActive projects context: {manager.active_projects}")

        # Test get_relevant_context_summary
        print("\n--- Testing get_relevant_context_summary ---")
        summary_for_new_query = manager.get_relevant_context_summary(
            current_query="User is now asking about performance optimization for the same script.",
            project_key="VSCode_DataProcessingScript"
        )
        print(summary_for_new_query)

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_jefe_context_test())
