from typing import Dict, Any, List

class DatasetValidatorError(Exception):
    """Raised when dataset validation fails."""

    def __init__(self, error: List):
        self.error = error
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return f"{self.error}"

    @staticmethod
    def create_error_json(error_type: str, **kwargs) -> Dict[str, Any]:
        return {
            "error_type": error_type,
            "details": kwargs,
        }

class SpecValidatorError(Exception):
    """Raised when contract validation fails."""

    def __init__(self, error_type: str, details: Dict[str, Any]):
        self.error_type = error_type
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return f"{self.error_type}: {self.details}"

    @staticmethod
    def create_error_json(error_type: str, **kwargs) -> Dict[str, Any]:
        return {
            "error_type": error_type,
            "details": kwargs,
        }