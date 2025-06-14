# multi-agent-system/src/tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, Optional # Added Optional

# To avoid circular imports, use TYPE_CHECKING for type hints of not-yet-fully-defined classes
# or classes that might import BaseTool. JefeContext will be defined in utils.
if TYPE_CHECKING:
    from ..utils.api_manager import APIManager # Assuming APIManager is in utils
        from ..utils.jefe_datatypes import JefeContext # JefeContext is now defined


class BaseTool(ABC):
    """
    Abstract base class for all tools that an agent (e.g., JefeAgent) can use.
    Tools are specialized components that perform specific tasks, often involving
    API calls or complex computations, based on the provided context.
    """

    def __init__(self, tool_name: str, tool_description: str, api_manager: 'Optional[APIManager]' = None):
        """
        Initializes the BaseTool.

        Args:
            tool_name: The unique name of the tool (e.g., "web_search", "code_interpreter").
            tool_description: A clear, concise description of what the tool does,
                              its expected inputs (via kwargs in execute), and what it returns.
                              This description is crucial for an LLM or a planner to decide
                              when and how to use the tool.
            api_manager: An optional instance of APIManager if the tool needs to make
                         external API calls.
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.api_manager = api_manager

        # It's good practice for tools to have their own logger.
        # This requires the logger utility to be easily importable.
        # For simplicity in this initial setup, we'll omit direct logger instantiation here,
        # assuming it could be passed or configured if needed by specific tools.
        # from ..utils.logger import get_logger
        # self.logger = get_logger(f"Tool.{self.tool_name}")
        # print(f"Tool '{self.tool_name}' initialized.") # Basic print for now

    @abstractmethod
    async def execute(self, context: 'JefeContext', **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the tool's main functionality.

        This method must be implemented by all concrete tool subclasses. It should
        perform the specific action the tool is designed for, using the provided
        context and any additional arguments.

        Args:
            context: An instance of `JefeContext` (or a similar context object)
                     providing situational awareness (e.g., screen content summary,
                     recent conversation, user profile). The exact structure of
                     `JefeContext` will be defined elsewhere.
            **kwargs: Additional keyword arguments that are specific to the tool's needs.
                      These should be documented in the tool's `tool_description`.

        Returns:
            A dictionary containing the results of the tool's execution.
            The structure should ideally include:
            - "status" (str): "success" or "error".
            - "data" (Any, optional): The primary output or result of the tool if successful.
            - "error" (str, optional): A message describing the error if status is "error".
            - "tool_used" (str): The name of the tool that was executed.
            Example: `{"status": "success", "data": "Search results...", "tool_used": "web_search"}`
        """
        pass # Subclasses must implement this

    def get_description(self) -> Dict[str, str]:
        """
        Returns a dictionary containing the tool's name and description.

        This is useful for providing tool information to an LLM or a planning
        component, enabling them to understand what the tool does and when to use it.

        Returns:
            A dictionary with "name" and "description" keys.
        """
        return {
            "name": self.tool_name,
            "description": self.tool_description
        }

if __name__ == '__main__':
    import asyncio # Required for the async main_tool_test

    # These are conceptual mocks for the purpose of illustrating the BaseTool structure.
    # In a real scenario, these would be properly defined and importable classes.
    if TYPE_CHECKING:
        class MockAPIManager: # Basic mock
            def __init__(self):
                print("MockAPIManager initialized for BaseTool test.")

        class MockJefeContext: # Basic mock
            def __init__(self, content: str = "Sample context"):
                self.content = content
                print(f"MockJefeContext initialized with content: '{content}'")
    else: # Define them as simple classes if not type checking for runtime example
        class MockAPIManager:
             def __init__(self):
                print("MockAPIManager initialized for BaseTool test.")
        class MockJefeContext:
            def __init__(self, content: str = "Sample context"):
                self.content = content
                print(f"MockJefeContext initialized with content: '{content}'")


    class MyExampleTool(BaseTool):
        """An example concrete implementation of BaseTool."""
        def __init__(self, api_manager: 'Optional[APIManager]' = None):
            super().__init__(
                tool_name="MyExampleTool",
                tool_description="This is an example tool that performs a conceptual task and returns a message. It accepts 'field1' in kwargs.",
                api_manager=api_manager
            )
            # If this specific tool needed a logger:
            # from src.utils.logger import get_logger # Adjust path as needed
            # self.logger = get_logger(f"Tool.{self.tool_name}")
            # self.logger.info(f"{self.tool_name} instance created.")


        async def execute(self, context: 'JefeContext', **kwargs: Any) -> Dict[str, Any]:
            """Executes the example tool's action."""
            # self.logger.info(f"Executing {self.tool_name}...")
            print(f"Executing {self.tool_name} with context type: {type(context)}, content: '{getattr(context, 'content', 'N/A')}'")

            field1_value = kwargs.get('field1', 'default_value')
            # Simulate some asynchronous work, like an API call
            await asyncio.sleep(0.05)

            output_data = f"Result from {self.tool_name}. Input field1 was: '{field1_value}'. Context content: '{getattr(context, 'content', 'N/A')}'"
            # self.logger.info(f"{self.tool_name} execution complete.")
            return {
                "status": "success",
                "data": output_data,
                "tool_used": self.tool_name
            }

    async def main_tool_test():
        """Main function to test the BaseTool structure and an example tool."""
        print("\n--- Running BaseTool Example Test ---")

        # Instantiate the mock APIManager (if your example tool needed it)
        mock_api_manager = MockAPIManager()

        # Instantiate the example tool
        example_tool = MyExampleTool(api_manager=mock_api_manager)

        # Get and print the tool's description (as an LLM might see it)
        print("\nTool Description:")
        print(example_tool.get_description())

        # Create a mock context
        mock_context_data = MockJefeContext(content="This is the current operational context for the tool.")

        # Execute the tool with some example kwargs
        print("\nExecuting tool...")
        execution_kwargs = {"field1": "custom_test_value"}
        result = await example_tool.execute(mock_context_data, **execution_kwargs)

        print("\nTool Execution Result:")
        print(result)

        # Example of a tool that might indicate an error
        class ErrorTool(BaseTool):
            def __init__(self):
                super().__init__("ErrorTool", "A tool that simulates an error.")
            async def execute(self, context: 'JefeContext', **kwargs: Any) -> Dict[str, Any]:
                return {"status": "error", "error": "Simulated error in ErrorTool", "tool_used": self.tool_name}

        error_tool_instance = ErrorTool()
        print("\nExecuting ErrorTool...")
        error_result = await error_tool_instance.execute(mock_context_data)
        print("ErrorTool Result:")
        print(error_result)


    # Running the async test function
    # Note: The `if __name__ == '__main__':` block for library code often contains
    # illustrative examples or simple tests. More comprehensive tests would typically
    # be in a separate test suite.
    if os.name == 'nt': # Optional: Windows specific policy for asyncio if needed
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_tool_test())
    print("\n--- BaseTool definition and example test finished ---")
