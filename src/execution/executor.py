from experiments.code import relationship_generation

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class Executor:
    def __init__(self):
        return
    
    def run_llm_code(self):
        return relationship_generation(1000, 42)
    
    #TODO Change to another file
    def visualize_data(self, results: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Visualize generated synthetic data.
        Assumes results = (x, y).
        """
        x, y = results

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Empty data passed to visualization")

        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, alpha=0.5, s=10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Synthetic Data Visualization")
        plt.tight_layout()
        plt.show()