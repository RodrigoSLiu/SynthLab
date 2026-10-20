from src.domain.contract import Contract
from src.primitives.registry import Registry

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
    
