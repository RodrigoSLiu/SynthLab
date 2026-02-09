from ollama import chat
from ollama import ChatResponse
from typing import Dict
import os
import yaml

from src.domain.contract import Contract
from src.llm.prompts import create_spec_prompt, create_yaml_error_prompt

class LLMError(Exception):
    pass


class OllamaClient():
    def __init__(self, model):
        self.model = model
        self.context = None
    
    async def generate_spec_from_intent(self, schema, intent:str) -> str:
        if not intent.strip():
            raise LLMError("User intent is empty")
        
        prompt = create_spec_prompt(schema, intent)
        response = await self._get_llm_response(prompt)

        return response
    
    async def regenerate_spec_from_error(self, error, invalid_yaml:str) -> str:
        if not (error or invalid_yaml):
            raise LLMError("Error or prompt not specified")
        
        prompt = create_yaml_error_prompt(error, invalid_yaml)
        print(prompt)
        response = await self._get_llm_response(prompt)

        return response

    async def generate_code(self, code_prompt:str) -> str:
        if not code_prompt.strip():
            raise LLMError("Prompt is empty")
        
        response = await self._get_llm_response(code_prompt)
        
        return response
    
    async def _get_llm_response(self, prompt):
        response: ChatResponse = chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': f'{prompt}',
            },
        ])
        if os.getenv("DEBUG"):
            print(f"User: {prompt[:50]}(...)")
            print(f"Ollama: {response['message']['content']}")

        return response['message']['content']