# multi-agent-system/src/utils/jefe_datatypes.py
from dataclasses import dataclass, field, asdict # Added asdict for potential to_dict method
from enum import Enum
from typing import List, Optional, Dict, Any # Added Dict, Any for to_dict

class JefeTaskType(Enum):
    """
    Enumeration of different task types that Jefe might identify or work on.
    This helps in categorizing tasks for specialized handling or routing.
    """
    IMMEDIATE_CODING_HELP = "immediate_coding_help"
    ARCHITECTURE_GUIDANCE = "architecture_guidance"
    DEBUGGING_ASSISTANCE = "debugging_assistance"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COMPLEX_PROBLEM_SOLVING = "complex_problem_solving"
    CODE_REVIEW = "code_review"
    KNOWLEDGE_QUERY = "knowledge_query" # For general questions
    # Add a default or unknown type if necessary for robustness
    UNKNOWN = "unknown_task_type"

    def __str__(self):
        """Returns the string value of the enum member."""
        return self.value

@dataclass
class JefeContext:
    """
    Represents the contextual information available to an agent (e.g., JefeAgent)
    at a specific moment in time. This context is crucial for the agent to make
    informed decisions and provide relevant assistance.
    """
    # Core contextual information
    screen_content: str # Textual content extracted from the user's screen.
    audio_transcript: str # Transcript of recent audio input (e.g., user speaking).

    # Optional, more specific contextual details
    current_ide: Optional[str] = None # Name of the Integrated Development Environment (IDE) being used (e.g., "VSCode", "PyCharm").
    programming_language: Optional[str] = None # Detected primary programming language in focus (e.g., "Python", "JavaScript").
    project_type: Optional[str] = None # Type of project the user is working on (e.g., "Web Application", "Data Analysis Script").

    # Lists for accumulating relevant information during an interaction or session
    error_messages: List[str] = field(default_factory=list) # Any error messages observed from IDE, linters, or compilers.
    previous_suggestions: List[str] = field(default_factory=list) # History of suggestions already provided to the user.
    # Could also include:
    # current_file_path: Optional[str] = None
    # cursor_position: Optional[Dict[str, int]] = None # e.g., {"line": 10, "column": 4}
    # selected_code_snippet: Optional[str] = None
    # relevant_documentation_links: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the dataclass instance to a dictionary."""
        return asdict(self)

    def summarize(self, max_screen_len: int = 500, max_audio_len: int = 300) -> str:
        """Provides a concise string summary of the context."""
        summary_parts = []
        if self.screen_content:
            summary_parts.append(f"Screen (first {max_screen_len} chars): {self.screen_content[:max_screen_len]}...")
        if self.audio_transcript:
            summary_parts.append(f"Audio (first {max_audio_len} chars): {self.audio_transcript[:max_audio_len]}...")
        if self.current_ide:
            summary_parts.append(f"IDE: {self.current_ide}")
        if self.programming_language:
            summary_parts.append(f"Language: {self.programming_language}")
        if self.error_messages:
            summary_parts.append(f"Recent Errors: {'; '.join(self.error_messages[-2:])}") # Show last 2 errors

        return "\n".join(summary_parts) if summary_parts else "No context available."


if __name__ == '__main__':
    print("--- JefeTaskType Enum Members ---")
    for task_type in JefeTaskType:
        print(f"{task_type.name}: {task_type.value} (str: {str(task_type)})")

    print("\n--- JefeContext Dataclass Example ---")
    example_context_empty = JefeContext(screen_content="", audio_transcript="")
    print(f"Empty context: {example_context_empty}")
    print(f"Empty context summary: {example_context_empty.summarize()}")


    example_context_filled = JefeContext(
        screen_content="class MyClass:\n  def __init__(self):\n    pass\n# ... more code ...",
        audio_transcript="User asked about Python classes and how to implement a constructor. They seemed unsure about self.",
        current_ide="VSCode",
        programming_language="Python",
        project_type="Web Application",
        error_messages=["SyntaxError: unexpected EOF while parsing", "IndentationError: expected an indented block"],
        previous_suggestions=["Consider adding a constructor with 'self'.", "Ensure your methods are indented correctly."]
    )
    print(f"\nFilled context: {example_context_filled}")
    print(f"Filled context (dict): {example_context_filled.to_dict()}")
    print(f"Filled context summary:\n{example_context_filled.summarize()}")
    print(f"Error messages from filled context: {example_context_filled.error_messages}")

    # Test default factory for lists to ensure they are distinct for different instances
    print("\n--- Testing default_factory for lists ---")
    ctx1 = JefeContext(screen_content="s1", audio_transcript="a1")
    ctx2 = JefeContext(screen_content="s2", audio_transcript="a2")

    ctx1.error_messages.append("Error specific to ctx1")
    ctx1.previous_suggestions.append("Suggestion for ctx1")

    print(f"ctx1 errors: {ctx1.error_messages}")
    print(f"ctx1 suggestions: {ctx1.previous_suggestions}")
    print(f"ctx2 errors: {ctx2.error_messages}")
    print(f"ctx2 suggestions: {ctx2.previous_suggestions}")

    assert ctx1.error_messages != ctx2.error_messages, "Error messages lists should be distinct"
    assert ctx1.previous_suggestions != ctx2.previous_suggestions, "Previous suggestions lists should be distinct"
    print("Default factory test for lists passed: Instances have independent list attributes.")
