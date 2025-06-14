# multi-agent-system/src/utils/jefe_task_analyzer.py
from typing import List, Dict, Any, Optional # Dict, Any might be needed for broader context handling if extended
from dataclasses import dataclass, field

from .jefe_datatypes import JefeContext, JefeTaskType # Assuming jefe_datatypes.py is in the same directory

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

@dataclass
class TaskAnalysis:
    """
    Represents the output of JefeTaskAnalyzer, detailing the nature
    and recommended handling of the current context.
    """
    task_type: JefeTaskType
    complexity: str  # e.g., "simple", "moderate", "complex"
    immediate_issues: List[str] = field(default_factory=list) # Specific errors or problems detected
    suggested_tools: List[str] = field(default_factory=list) # List of tool names that might be useful
    priority: str # e.g., "high", "medium", "low" based on urgency or error presence


class JefeTaskAnalyzer:
    """
    Analyzes incoming JefeContext to determine task type, complexity,
    immediate issues, and suggests an appropriate response strategy or tools.
    This analyzer is specific to the needs of a "Jefe" type coding assistant.
    """
    def __init__(self):
        self.logger = get_logger("JefeTaskAnalyzer")

        # Predefined error and performance patterns for _detect_immediate_issues
        # These are case-insensitive checks.
        self.error_patterns = [
            'syntaxerror', 'typeerror', 'referenceerror', 'nameerror', 'attributeerror',
            'indexerror', 'keyerror', 'valueerror', 'zerodivisionerror',
            'compilation failed', 'build failed', 'test failed',
            'undefined', 'null pointer', 'segmentation fault', 'exception', 'error:',
            'failed to compile', 'linker error', 'unresolved external symbol'
        ]
        self.performance_patterns = [
            'memory leak', 'slow query', 'timeout', 'bottleneck',
            'high cpu usage', 'infinite loop', 'excessive redraw', 'lagging', 'unresponsive'
        ]
        # Keywords to assist in task type classification from combined screen/audio context
        self.task_type_keywords = {
            JefeTaskType.DEBUGGING_ASSISTANCE: ['debug', 'fix', 'traceback', 'problem with this code', 'not working', 'what went wrong'],
            JefeTaskType.ARCHITECTURE_GUIDANCE: ['architecture', 'design pattern', 'structure my app', 'refactor this module', 'system design', 'component interaction', 'best way to build'],
            JefeTaskType.PERFORMANCE_OPTIMIZATION: ['performance', 'optimize', 'slow', 'speed up', 'profile code', 'memory usage'],
            JefeTaskType.IMMEDIATE_CODING_HELP: ['how do i', 'implement function', 'write code for', 'example of', 'syntax for', 'code snippet for'],
            JefeTaskType.COMPLEX_PROBLEM_SOLVING: ['complex problem', 'research topic', 'investigate issue', 'deep dive into', 'multi-step task'],
            JefeTaskType.CODE_REVIEW: ['review this code', 'critique my code', 'code feedback', 'suggestions for this code'],
            JefeTaskType.KNOWLEDGE_QUERY: ['what is', 'explain concept', 'tell me about', 'how does x work']
        }
        self.logger.info("JefeTaskAnalyzer initialized with error, performance, and task keyword patterns.")


    def analyze_context(self, context: JefeContext) -> TaskAnalysis:
        """
        Analyzes the given JefeContext and returns a TaskAnalysis object.
        This is the main public method of the analyzer.
        """
        self.logger.debug(f"Analyzing context. Screen content (first 100 chars): '{context.screen_content[:100]}...', Audio transcript (first 100 chars): '{context.audio_transcript[:100]}...'")

        immediate_issues = self._detect_immediate_issues(context.screen_content, context.error_messages)
        complexity_indicators = self._assess_complexity(context)
        complexity_level = self._determine_complexity_level(complexity_indicators, immediate_issues)
        task_type = self._classify_task_type(context, immediate_issues)
        suggested_tools = self._recommend_tools(task_type, complexity_level, immediate_issues)

        # Determine priority: High if critical issues, complex tasks, or explicit user urgency.
        priority = "normal" # Default priority
        if immediate_issues and any(err_patt in " ".join(immediate_issues).lower() for err_patt in self.error_patterns):
            priority = "high" # Errors usually mean high priority
        elif complexity_level == "complex":
            priority = "high" # Complex tasks might also be high priority
        elif "urgent" in context.audio_transcript.lower() or "asap" in context.audio_transcript.lower():
            priority = "high"

        analysis_result = TaskAnalysis(
            task_type=task_type,
            complexity=complexity_level,
            immediate_issues=immediate_issues,
            suggested_tools=suggested_tools,
            priority=priority
        )
        self.logger.info(f"Context analysis complete: Type='{task_type.value}', Complexity='{complexity_level}', Issues='{len(immediate_issues)}', Tools='{suggested_tools}', Priority='{priority}'")
        return analysis_result

    def _detect_immediate_issues(self, screen_content: str, error_messages_from_context: List[str]) -> List[str]:
        """
        Detects immediate issues like errors or performance problems from screen content and explicit error messages.
        """
        issues: List[str] = []
        lower_screen_content = screen_content.lower()

        # Check for error patterns in screen content
        for pattern in self.error_patterns:
            if pattern in lower_screen_content: # error_patterns are already lowercased in init
                issues.append(f"Potential error indicator: '{pattern}' found in screen content.")

        # Add explicit error messages from context and check them against patterns
        for error_msg in error_messages_from_context:
            issues.append(f"Explicit error message: '{error_msg}'")
            # Check patterns again in explicit error messages for more specific matches
            for pattern in self.error_patterns:
                if pattern in error_msg.lower() and f"Error message confirms: '{pattern}'" not in " ".join(issues): # Avoid duplicate type of message
                    issues.append(f"Error message confirms: '{pattern}'.")

        # Check for performance patterns in screen content
        for pattern in self.performance_patterns:
            if pattern in lower_screen_content: # performance_patterns are already lowercased in init
                issues.append(f"Potential performance indicator: '{pattern}' found in screen content.")

        # Remove duplicates while trying to preserve order (dict.fromkeys is Python 3.7+)
        unique_issues = list(dict.fromkeys(issues))
        if unique_issues:
             self.logger.debug(f"Detected {len(unique_issues)} immediate issues: {unique_issues}")
        return unique_issues


    def _assess_complexity(self, context: JefeContext) -> Dict[str, int]:
        """
        Assesses various indicators of complexity from the context.
        Returns a dictionary of these indicators.
        """
        # Basic heuristic: count import statements or similar as a proxy for multi-file interaction.
        multiple_files_hint = len([line for line in context.screen_content.split('\n')
                                     if 'import ' in line or 'require(' in line or 'from .' in line or 'include <' in line])

        # Count of technical terms can indicate domain complexity.
        technical_terms = [
            'async', 'await', 'promise', 'callback', 'api', 'database', 'pointer', 'memory',
            'thread', 'mutex', 'class', 'interface', 'module', 'lambda', 'decorator',
            'generator', 'virtualization', 'container', 'microservice', 'queue', 'stream',
            'recursive', 'algorithm', 'data structure', 'encryption', 'authentication',
            'framework', 'library', 'sdk', 'compiler', 'interpreter', 'polymorphism',
            'inheritance', 'encapsulation', 'abstract', 'static', 'dynamic'
        ]
        technical_terms_count = sum(1 for word in context.screen_content.lower().split() if word in technical_terms)

        indicators = {
            'multiple_files_hint': multiple_files_hint,
            'error_count': len(context.error_messages or []),
            'audio_questions_count': (context.audio_transcript or '').lower().count('?'),
            'code_length_lines': len(context.screen_content.split('\n')),
            'technical_terms_count': technical_terms_count
        }
        self.logger.debug(f"Complexity indicators assessed: {indicators}")
        return indicators

    def _determine_complexity_level(self, indicators: Dict[str, int], immediate_issues: List[str]) -> str:
        """
        Determines a qualitative complexity level ("simple", "moderate", "complex")
        based on the assessed indicators and presence of immediate issues.
        """
        score = 0
        score += indicators.get('multiple_files_hint', 0) * 1      # Each hint of multiple files adds 1
        score += indicators.get('error_count', 0) * 2              # Each explicit error adds 2
        score += indicators.get('audio_questions_count', 0) * 1    # Each question in audio adds 1
        score += min(indicators.get('code_length_lines', 0) // 100, 5) * 2 # 2 points per 100 lines of code, capped at 10 points
        score += min(indicators.get('technical_terms_count', 0) // 2, 6) * 1 # 1 point per 2 tech terms, capped at 6 points

        if len(immediate_issues) >= 2: # Two or more distinct issues significantly increase complexity
            score += 3
        elif immediate_issues: # At least one issue
            score += 1

        self.logger.debug(f"Calculated complexity score: {score}")
        # Thresholds for complexity levels
        if score <= 5:
            return "simple"
        elif score <= 12:
            return "moderate"
        else:
            return "complex"

    def _classify_task_type(self, context: JefeContext, immediate_issues: List[str]) -> JefeTaskType:
        """
        Classifies the primary task type based on context and detected issues.
        Prioritizes debugging if errors are present.
        """
        lower_screen = context.screen_content.lower()
        lower_audio = context.audio_transcript.lower()
        # Combine screen and audio for keyword search, giving more weight or context to screen.
        # A more advanced approach might analyze them separately or use embeddings.
        combined_text_for_keywords = lower_screen + " " + lower_audio + " " + " ".join(context.error_messages).lower()


        # Prioritize debugging or performance if specific issues are detected
        if immediate_issues:
            for issue in immediate_issues:
                issue_lower = issue.lower()
                if any(err_patt in issue_lower for err_patt in self.error_patterns):
                    self.logger.debug(f"Classifying as DEBUGGING_ASSISTANCE due to immediate error patterns in: '{issue_lower}'")
                    return JefeTaskType.DEBUGGING_ASSISTANCE
                if any(perf_patt in issue_lower for perf_patt in self.performance_patterns):
                     self.logger.debug(f"Classifying as PERFORMANCE_OPTIMIZATION due to immediate performance patterns in: '{issue_lower}'")
                     return JefeTaskType.PERFORMANCE_OPTIMIZATION

        # Keyword-based classification for other task types
        # Iterate in a defined order of precedence if necessary (e.g., more specific types first)
        # For now, first match wins after error/perf check.
        for task_type_enum, keywords in self.task_type_keywords.items():
            if task_type_enum in [JefeTaskType.DEBUGGING_ASSISTANCE, JefeTaskType.PERFORMANCE_OPTIMIZATION]:
                continue # Already handled if issues were present
            if any(keyword in combined_text_for_keywords for keyword in keywords):
                self.logger.debug(f"Classifying as {task_type_enum.value} due to keywords: {keywords}")
                return task_type_enum

        # Default classification if no specific keywords match strongly
        if context.programming_language or "code" in lower_screen or "function" in lower_screen or "class" in lower_screen:
            self.logger.debug("Defaulting to IMMEDIATE_CODING_HELP due to general coding context indications.")
            return JefeTaskType.IMMEDIATE_CODING_HELP

        self.logger.debug("Defaulting to UNKNOWN task type as no specific keywords or coding context matched strongly.")
        return JefeTaskType.UNKNOWN

    def _recommend_tools(self, task_type: JefeTaskType, complexity: str, immediate_issues: List[str]) -> List[str]:
        """
        Recommends a list of tool names based on task type, complexity, and issues.
        This is a placeholder for future tool integration.
        """
        tools: List[str] = []
        # Basic tool recommendation logic (to be expanded when tools are implemented)
        if task_type == JefeTaskType.DEBUGGING_ASSISTANCE or any(err_patt in " ".join(immediate_issues).lower() for err_patt in self.error_patterns):
            tools.append("error_explainer_tool") # Explains common errors
            tools.append("code_analyzer_tool")   # Static analysis for potential bugs
            if complexity == "complex":
                tools.append("interactive_debugger_tool") # Hypothetical

        elif task_type == JefeTaskType.IMMEDIATE_CODING_HELP:
            tools.append("code_snippet_generator_tool")
            tools.append("documentation_search_tool")

        elif task_type == JefeTaskType.ARCHITECTURE_GUIDANCE:
            tools.append("design_pattern_advisor_tool")
            tools.append("project_structure_tool")

        elif task_type == JefeTaskType.PERFORMANCE_OPTIMIZATION:
            tools.append("code_profiler_tool")
            tools.append("optimization_suggestion_tool")

        elif task_type == JefeTaskType.CODE_REVIEW:
            tools.append("code_analyzer_tool")
            tools.append("style_checker_tool")
            tools.append("best_practices_tool")

        elif task_type == JefeTaskType.KNOWLEDGE_QUERY:
            tools.append("web_search_tool") # For general knowledge
            tools.append("documentation_search_tool") # If context suggests technical docs

        elif task_type == JefeTaskType.COMPLEX_PROBLEM_SOLVING:
            tools.extend(["web_search_tool", "code_analyzer_tool", "documentation_search_tool"])

        # Add a general-purpose tool if complexity is high or no specific tools fit
        if complexity == "complex" and not tools:
            tools.append("advanced_reasoning_tool") # A powerful, general LLM call

        # Remove duplicates while preserving order
        recommended_tools = list(dict.fromkeys(tools))
        self.logger.debug(f"Recommended tools for task type '{task_type.value}' (complexity '{complexity}'): {recommended_tools}")
        return recommended_tools

if __name__ == '__main__':
    import asyncio # Required for main_jefe_context_test which might be adapted

    analyzer = JefeTaskAnalyzer()

    test_contexts = [
        JefeContext(screen_content="SyntaxError: invalid syntax on line 10 of my_script.py",
                    audio_transcript="User: Argh, I have an error, can you help me fix this syntax error?",
                    error_messages=["SyntaxError: invalid syntax on line 10"],
                    programming_language="Python"),
        JefeContext(screen_content="class User { constructor() { console.log('user created'); } }",
                    audio_transcript="User: How do I write a class in JavaScript to represent a user?",
                    programming_language="JavaScript"),
        JefeContext(screen_content="This Python code is very slow when processing large datasets. It uses pandas.",
                    audio_transcript="User: My application is lagging badly, can you help optimize this Python script?",
                    project_type="Data Processing", programming_language="Python"),
        JefeContext(screen_content="Thinking about building a new microservice for auth using FastAPI.",
                    audio_transcript="User: What's the best way to design an authentication microservice for my system?",
                    current_ide="VSCode", programming_language="Python"),
        JefeContext(screen_content="Large codebase with multiple modules, imports, and async functions... \n async function process_data() {...}",
                    audio_transcript="User: I need to understand this complex system and then refactor its main data processing pipeline. It's really complicated.",
                    programming_language="Python",
                    # Simulating complexity indicators:
                    error_messages=["Warning: Deprecated API used", "Potential race condition detected"],
                    previous_suggestions=["Consider using a queue for tasks."]),
        JefeContext(screen_content="def my_func(a,b,c):\n  return a+b+c\nprint(my_func(1,2,3))",
                    audio_transcript="User: Please review this Python code for me.",
                    programming_language="Python"),
        JefeContext(screen_content="Kubernetes deployment YAML file showing multiple replicas.",
                    audio_transcript="User: Explain Kubernetes replica sets to me.",
                    project_type="DevOps")
    ]

    for i, ctx in enumerate(test_contexts):
        print(f"\n--- Test Context {i+1} ---")
        print(f"  Screen (first 70 chars): '{ctx.screen_content[:70]}...'")
        print(f"  Audio (first 70 chars): '{ctx.audio_transcript[:70]}...'")
        if ctx.error_messages:
            print(f"  Explicit Errors: {ctx.error_messages}")

        analysis = analyzer.analyze_context(ctx)
        print(f"  Analysis Result:")
        print(f"    Task Type: {analysis.task_type.value}") # Use .value for cleaner print
        print(f"    Complexity: {analysis.complexity}")
        print(f"    Immediate Issues: {analysis.immediate_issues if analysis.immediate_issues else 'None'}")
        print(f"    Suggested Tools: {analysis.suggested_tools if analysis.suggested_tools else 'None'}")
        print(f"    Priority: {analysis.priority}")
        print("-" * 30)
