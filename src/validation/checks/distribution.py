from typing import Dict, Any
import numpy as np
from scipy import stats


def run_ks_test(
    data: np.ndarray,
    distribution: str,
    params: Dict[str, float],
    alpha: float,
) -> Dict[str, Any]:
    """
    Run a Kolmogorov–Smirnov test against a declared distribution.
    """
    if data.ndim != 1:
        raise ValueError("KS test requires a 1D array")

    if distribution == "uniform":
        loc = params["min"]
        scale = params["max"] - params["min"]
        cdf = stats.uniform(loc=loc, scale=scale).cdf

    elif distribution == "normal":
        loc = params["mean"]
        scale = params["std"]
        cdf = stats.norm(loc=loc, scale=scale).cdf

    else:
        raise ValueError(f"Unsupported distribution for KS test: {distribution}")

    statistic, p_value = stats.kstest(data, cdf)

    return {
        "check": "ks",
        "distribution": distribution,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "passed": p_value >= alpha,
    }