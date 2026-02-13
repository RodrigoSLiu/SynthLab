from ollama import chat
from ollama import ChatResponse
from typing import Dict
import logging
import yaml

from src.domain.contract import Contract
from src.llm.prompts import create_spec_prompt, create_yaml_error_prompt, create_code_prompt

logger = logging.getLogger(__name__)

class LLMError(Exception):
    pass


class OllamaClient():
    def __init__(self, spec_model, code_model):
        self.spec_model = spec_model
        self.code_model = code_model
        self.context = None
    
    async def generate_spec_from_intent(self, schema, intent:str) -> str:
        if not intent.strip():
            raise LLMError("User intent is empty")
        
        prompt = create_spec_prompt(schema, intent)
        response = await self._get_llm_response(self.spec_model, prompt)

        return response
    
    async def regenerate_spec_from_error(self, error, invalid_yaml:str) -> str:
        if not (error or invalid_yaml):
            raise LLMError("Error or prompt not specified")
        
        prompt = create_yaml_error_prompt(error, invalid_yaml)
        response = await self._get_llm_response(self.spec_model, prompt)

        return response

    async def generate_code_from_spec(self, contract: Contract) -> str:
        if not contract:
            raise LLMError("Contract is missing or not valid")
        
        prompt = create_code_prompt(contract)
        response = await self._get_llm_response(self.code_model, prompt)
        
        return response
    
    async def _get_llm_response(self, chosen_model, prompt):
        response: ChatResponse = chat(model=chosen_model, messages=[
            {
                'role': 'user',
                'content': f'{prompt}',
            },
        ])
        logger.debug("#### USER ####")
        logger.debug("User: %s(...)", prompt)
        logger.debug("############")
        logger.debug("#### OLLAMA ####")
        logger.debug("Ollama: %s", response['message']['content'])
        logger.debug("############")

        return response['message']['content']