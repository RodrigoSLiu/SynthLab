from typing import Dict, Any, List
from src.errors import ContractError

import copy


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
    
    def _validate_semantics(self):
        variables = self.variables
        marginals = self.marginals
        relationships = self.relationships

        for marg in marginals:
            var = marg["variable"]

            if var not in variables:
                raise ContractError(
                    error_type="UNKNOWN_VARIABLE_IN_MARGINAL",
                    details={
                        "variable": var,
                        "available_variables": list(variables.keys())
                    }
                )

        for rel in relationships:
            target = rel["target"]

            if target not in variables:
                raise ContractError(
                    "UNKNOWN_TARGET_VARIABLE",
                    {"variable": target}
                )

    def get_variables(self):
        return copy.deepcopy(self.variables)

    def get_marginals(self):
        return copy.deepcopy(self.marginals)

    def get_relationships(self):
        return copy.deepcopy(self.relationships)

    def get_validation(self):
        return copy.deepcopy(self.validation)
