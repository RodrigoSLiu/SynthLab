from experiments.code import relationship_generation
from src.domain.contract import Contract
from src.primitives.registry import Registry
from src.core.errors import ContractError

from itertools import combinations
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd



class Executor:
    def __init__(self):
        self.registry = Registry()
    

    def execute_spec(self, contract: Contract, num_samples: int, seed: int):
        variables = contract.get_variables().keys()
        marginals = contract.get_marginals()
        relationships = contract.get_relationships()

        registry = Registry()
        dataset = {}

        dataset_marginals = self._generate_marginals(marginals, num_samples, seed)
        dataset.update(dataset_marginals)
        dataset_relationships = self._generate_relationships(dataset, relationships)

        return pd.DataFrame(dataset)


    def _generate_marginals(self, marginals: List[Dict[str, Any]], num_samples: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)

        results = {}

        for marg in marginals:
            variable = marg["variable"]
            generator_spec = marg["generator"]

            primitive = self.registry.get_distribution(generator_spec["name"])
            array = primitive.generate(
                rng=rng,
                size=num_samples,
                params=generator_spec["params"]
            )

            results[variable] = array

        return results


    def _generate_relationships(self, dataset: Dict[str, np.ndarray], relationships: List[Dict[str, Any]]):
        results = {}

        for rel in relationships:
            target = rel["target"]
            input_names = rel["inputs"]
            function_spec = rel["function"]

            primitive = self.registry.get_relationship(function_spec["name"])

            input_arrays = {
                name: dataset[name] for name in input_names
            }

            y = primitive.generate(
                input_arrays,
                function_spec["params"]
            )

            dataset[target] = y

        return results


    @staticmethod
    def execute_generated_code(code: str, num_samples: int, seed: int):
        namespace = {}

        exec(code, namespace)

        if "relationship_generation" not in namespace:
            raise RuntimeError("Generated code did not define relationship_generation")

        func = namespace["relationship_generation"]

        return func(num_samples, seed)
    

    #TODO Change to another file
    def visualize_data(self, df: pd.DataFrame) -> None:
        """
        Generic visualization for synthetic datasets.

        - 1 column  -> histogram
        - 2 columns -> scatter plot
        - >2 columns -> pairwise scatter plots
        """

        if df is None or df.empty:
            raise ValueError("Empty DataFrame passed to visualization")

        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            raise ValueError("No numeric columns available for visualization")

        n_cols = numeric_df.shape[1]

        # --- Single variable ---
        if n_cols == 1:
            col = numeric_df.columns[0]
            plt.figure(figsize=(6, 4))
            plt.hist(numeric_df[col], bins=50)
            plt.xlabel(col)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.show()
            return

        # --- Two variables ---
        if n_cols == 2:
            col1, col2 = numeric_df.columns
            plt.figure(figsize=(6, 4))
            plt.scatter(numeric_df[col1], numeric_df[col2], s=10, alpha=0.5)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f"{col2} vs {col1}")
            plt.tight_layout()
            plt.show()
            return

        # --- More than two variables ---
        pairs = list(combinations(numeric_df.columns, 2))

        for col1, col2 in pairs:
            plt.figure(figsize=(6, 4))
            plt.scatter(numeric_df[col1], numeric_df[col2], s=10, alpha=0.5)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f"{col2} vs {col1}")
            plt.tight_layout()
            plt.show()