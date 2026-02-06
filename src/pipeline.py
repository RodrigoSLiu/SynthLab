from src.llm.ollama_client import OllamaClient
from src.domain.contract import Contract
from src.validation.validator import Validator, ValidationError
from src.utils.utils import write_generated_code, load_schema, extract_yaml, write_generated_yaml
from src.utils.llm_utils import handle_validation_error
from dotenv import load_dotenv
from src.llm.prompts import create_spec_prompt, create_code_prompt
from src.execution.executor import Executor

import os
import asyncio


async def run_pipeline():
    schema = load_schema("linear_regression.schema.json")
    validator = Validator(schema)

    model_name = os.getenv("LLM_MODEL")
    llm = OllamaClient(model_name)

    executor = Executor()

    intent = (
        "Generate a dataset with x uniformly distributed between 0 and 1 "
        "and y following y = 2x with additive Gaussian noise. "
        "Validate the distribution of x and the slope of the relationship."
    )

    if not os.getenv("DEBUG"):
        llm_spec = await llm.generate_spec_from_intent(schema, intent)
        yaml = extract_yaml(llm_spec)
    
    if not os.getenv("DEBUG"):
        print("########################")
        print(yaml)
        print("########################")

        try:
            spec = validator.validate_yaml(yaml)
            
            if spec:
                write_generated_yaml(spec, "specs/test.yaml") 
                contract = Contract.from_spec(spec)
        except ValidationError as e:
            handle_validation_error(e)
            return

        prompt = create_code_prompt(contract)
        code = await llm.generate_code(prompt)

        write_generated_code(code, "experiments/code.py")
    
    if os.getenv("DEBUG"):
        results = executor.run_llm_code()
        executor.visualize_data(results)

    return

async def main():
    load_dotenv()
    await run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())