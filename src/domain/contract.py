from typing import Dict, Any, List


class Contract:
    """
    Trusted internal representation of a statistical data-generation contract.
    Constructed ONLY from a schema-validated spec.
    """

    def __init__(
        self,
        variables: Dict[str, Dict[str, Any]],
        marginals: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        validation: Dict[str, Any],
    ):
        self.variables = variables
        self.marginals = marginals
        self.relationships = relationships
        self.validation = validation

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "Contract":
        return cls(
            variables=spec["variables"],
            marginals=spec["assumptions"]["marginals"],
            relationships=spec["relationships"],
            validation=spec["validation"],
        )