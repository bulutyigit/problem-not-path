"""Custom exceptions used throughout ReasonBench."""


class ReasonBenchError(RuntimeError):
    """Base exception for project-specific failures."""


class ConfigurationError(ReasonBenchError):
    """Raised when an experiment configuration is invalid."""


class PhaseGateError(ReasonBenchError):
    """Raised when a notebook attempts to bypass a phase gate."""


class VerificationError(ReasonBenchError):
    """Raised when an answer cannot be verified safely."""


class InstrumentationError(ReasonBenchError):
    """Raised when model signals cannot be captured or aligned."""


class StorageError(ReasonBenchError):
    """Raised when a durable artifact cannot be written or validated."""
