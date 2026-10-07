from enum import Enum


class TaskType(str, Enum):
    GENERAL = "general"
    FAST = "fast"
    REASONING = "reasoning"
    CODE = "code"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    AUTO = "auto"
    MULTIMODAL = "multimodal"
    LONG_CONTEXT = "long_context"
