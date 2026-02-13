import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from scipy.stats import linregress, kstest

logger = logging.getLogger(__name__)


class DatasetValidatorError(Exception):
    pass


class DatasetValidator:
    def __init__(self, contract):
        self.contract = contract

    def validate_dataset(self, dataset: pd.DataFrame) -> bool:
        errors = []

        # Validate variables
        try:
            self.validate_variables(dataset, self.contract.variables)
            logger.info("Variables validated")
        except DatasetValidatorError as e:
            errors.extend(e.args[0])

        # Validate marginal distribution (MVP: first marginal only)
        marginal = self.contract.marginals[0]
        column = marginal["variable"]
        dist_name = marginal["distribution"]
        params = marginal["params"]

        try:
            self.validate_distribution(
                dataset[column].to_numpy(),
                dist_name,
                params,
                self.contract.validation["distribution_test"],
            )
            logger.info("Distribution validated")
        except DatasetValidatorError as e:
            errors.extend(e.args[0])

        # Validate relationship (MVP: linear only)
        relationship = self.contract.relationships[0]

        try:
            self.validate_relationship(
                dataset,
                relationship,
                self.contract.validation["relationship_tolerance"],
            )
            logger.info("Relationship validated")
        except DatasetValidatorError as e:
            errors.extend(e.args[0])

        if errors:
            raise DatasetValidatorError(errors)

        return True

    def validate_variables(self, dataset: pd.DataFrame, contract_variables: dict) -> bool:
        errors = []

        if set(dataset.columns) != set(contract_variables.keys()):
            errors.append({
                "type": "COLUMN_MISMATCH",
                "expected": list(contract_variables.keys()),
                "actual": list(dataset.columns),
            })

        for col, spec in contract_variables.items():
            if spec["type"] == "float":
                if not np.issubdtype(dataset[col].dtype, np.floating):
                    errors.append({
                        "type": "DTYPE_MISMATCH",
                        "column": col,
                        "expected": "float",
                        "actual": str(dataset[col].dtype),
                    })

        if errors:
            raise DatasetValidatorError(errors)

        return True
    
    def validate_distribution(
        self,
        array: np.ndarray,
        distribution: str,
        params: Dict[str, Any],
        validation_config: Dict[str, Any],
    ) -> bool:

        errors = []

        if validation_config["test"] != "ks":
            return True  # MVP supports only ks

        alpha = validation_config["alpha"]

        if distribution == "uniform":
            min_val = params["min"]
            max_val = params["max"]

            # scale to 0-1 for KS test
            scaled = (array - min_val) / (max_val - min_val)

            statistic, p_value = kstest(scaled, "uniform")

            if p_value < alpha:
                errors.append({
                    "type": "KS_TEST_FAILED",
                    "p_value": float(p_value),
                    "alpha": alpha,
                })

        else:
            errors.append({
                "type": "UNSUPPORTED_DISTRIBUTION",
                "distribution": distribution,
            })

        if errors:
            raise DatasetValidatorError(errors)

        return True

    def validate_relationship(
        self,
        dataset: pd.DataFrame,
        relationship: Dict[str, Any],
        tolerance_config: Dict[str, Any],
    ) -> bool:

        errors = []

        if relationship["type"] != "linear":
            return True  # MVP only linear

        x = dataset[relationship["independent"]].to_numpy()
        y = dataset[relationship["dependent"]].to_numpy()

        slope_expected = relationship["slope"]
        tolerance = tolerance_config["slope"]

        slope, intercept, r_value, p_value, stderr = linregress(x, y)

        logger.debug(f"Observed slope: {slope}")

        if abs(slope - slope_expected) > tolerance:
            errors.append({
                "type": "SLOPE_MISMATCH",
                "expected": slope_expected,
                "observed": slope,
                "tolerance": tolerance,
            })

        if errors:
            raise DatasetValidatorError(errors)

        return True