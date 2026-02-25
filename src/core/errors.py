from typing import Dict, Any, List, Optional


class SynthLabError(Exception):
    """
    Base exception for the project.
    All custom exceptions should inherit from this.
    """

    def __init__(
        self,
        error_type: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return f"{self.error_type}: {self.details}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "details": self.details,
        }


class DatasetValidatorError(SynthLabError):
    """Raised when dataset validation fails."""

    def __init__(self, errors: List[Dict[str, Any]]):
        super().__init__(
            error_type="DATASET_VALIDATION_ERROR",
            details={"violations": errors},
        )
        self.errors = errors


class SpecValidatorError(SynthLabError):
    """Raised when YAML or schema validation fails."""

    def __init__(self, error_type: str, details: Dict[str, Any]):
        super().__init__(error_type=error_type, details=details)


class ContractError(SynthLabError):
    """Raised when semantic contract validation fails."""

    def __init__(self, error_type: str, details: Dict[str, Any]):
        super().__init__(error_type=error_type, details=details)