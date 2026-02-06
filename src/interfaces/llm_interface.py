from typing import Protocol


class LLMInterface(Protocol):
    def __init__(self, model):
        ...

    async def generate_code(self, intent:str, contract:dict) -> str:
        ...
    
    async def _construct_prompt(self, prompt:str) -> str:
        ...
