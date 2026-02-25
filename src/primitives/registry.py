from src.primitives.distributions import DistributionPrimitive, Normal, Uniform, Exponential, Pareto, Beta, LogNormal, Poisson, Binomial
from src.primitives.relationships import RelationshipPrimitive, Linear, Polynomial, Logistic, Interaction, PiecewiseLinear
from src.core.errors import ContractError


class Registry:
    def __init__(self):
        self._distribution_registry = {
            "normal": Normal,
            "uniform": Uniform,
            "exponential": Exponential,
            "pareto": Pareto,
            "beta": Beta,
            "lognormal": LogNormal,
            "poisson": Poisson,
            "binomial": Binomial,

        }

        self._relationship_registry = {
            "linear": Linear,
            "polynomial": Polynomial,
            "logistic": Logistic,
            "interaction": Interaction,
            "piecewise_linear": PiecewiseLinear
        }

    def get_distribution(self, name: str) -> DistributionPrimitive:
        if name not in self._distribution_registry:
            raise ContractError(
                "UNKNOWN_DISTRIBUTION",
                {"name": name}
            )
        
        return self._distribution_registry[name]()

    def get_relationship(self, name: str) -> RelationshipPrimitive:
        if name not in self._relationship_registry:
             raise ContractError(
                "UNKNOWN_DISTRIBUTION",
                {"name": name}
            )

        return self._relationship_registry[name]()  # instantiate here
    