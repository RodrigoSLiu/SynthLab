from experiments.code import relationship_generation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple


class Executor:
    def __init__(self):
        return
    
    @staticmethod
    def execute_generated_code(code: str, num_samples: int, seed: int):
        namespace = {}

        exec(code, namespace)

        if "relationship_generation" not in namespace:
            raise RuntimeError("Generated code did not define relationship_generation")

        func = namespace["relationship_generation"]

        return func(num_samples, seed)
    
    #TODO Change to another file
    def visualize_data(self, df) -> None:
        """
        Visualize generated synthetic data.
        Assumes df contains columns 'x' and 'y'.
        """
        if df.empty:
            raise ValueError("Empty DataFrame passed to visualization")

        required_columns = {"x", "y"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns {required_columns}")

        plt.figure(figsize=(6, 4))
        plt.scatter(df["x"], df["y"], alpha=0.5, s=10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Synthetic Data Visualization")
        plt.tight_layout()
        plt.show()