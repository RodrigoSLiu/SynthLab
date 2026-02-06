from typing import Dict
from typing import Protocol


class Validator(Protocol):
    def get_contract_validation(contract:str) -> Dict[str,any]:
        ...
    
    def validate(
        self, 
        code: str, 
        language: str,
        specification: str
    ) -> bool:
        ...
    