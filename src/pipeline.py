from src.llm.ollama_client import OllamaClient
from src.domain.contract import Contract
from src.validation.spec_validator import SpecValidator
from src.validation.dataset_validator import DatasetValidator
from src.validation.errors import SpecValidatorError
from src.utils.utils import write_generated_code, load_schema, extract_content_from_response, write_generated_yaml, build_dataframe, save_dataset_csv, load_dataset_csv
from src.llm.prompts import create_code_prompt
from src.execution.executor import Executor

from dotenv import load_dotenv
from typing import Optional, Dict, Any
import json
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

async def get_spec_with_retry(
    llm,
    validator: SpecValidator,
    schema: Dict[str, Any],
    intent: str,
    retry: int = 3,
) -> Dict[str, Any]:

    yaml_text: str = ""
    last_error: Optional[SpecValidatorError] = None

    for i in range(retry):
        logger.info("Trying %d/%d...", i + 1, retry)

        if last_error is not None:
            logger.warning("YAML validation failed, re-running...")
            llm_spec = await llm.regenerate_spec_from_error(last_error, yaml_text)
        else:
            llm_spec = await llm.generate_spec_from_intent(schema, intent)

        try:
            logger.info("#### EXTRACTED YAML #####")
            yaml_text = extract_content_from_response(llm_spec)
            logger.info(yaml_text)

            logger.info("#### SPEC VALIDATION #####")
            spec = validator.validate_yaml(yaml_text)

            return spec

        except SpecValidatorError as e:
            last_error = e
            logger.error("Validator raised error: %s", e)

    # If we reach here, all retries failed
    raise SpecValidatorError(
        error_type="SPEC_GENERATION_FAILED",
        details={
            "message": f"Failed after {retry} attempts",
            "last_error_type": last_error.error_type if last_error else None,
            "last_error_details": last_error.details if last_error else None,
            "last_yaml_attempt": yaml_text,
        },
    )


async def run_pipeline():
    schema = load_schema("linear_regression.schema.json")
    spec_validator = SpecValidator(schema)

    spec_model_name = os.getenv("SPEC_LLM_MODEL")
    code_model_name = os.getenv("CODE_LLM_MODEL")
    llm = OllamaClient(spec_model_name, code_model_name)

    executor = Executor()

    intent = (
        "Generate a dataset with x uniformly distributed between 0 and 1"
        "and y following y = 2x with additive Gaussian noise."
        "Validate the distribution of x and the slope of the relationship."
    )

    spec = await get_spec_with_retry(llm, spec_validator, schema, intent, retry=3)
    try:
        logger.info("Writing yaml...")
        write_generated_yaml(spec, "specs/test.yaml")
        contract = Contract.from_spec(spec)
    except ValueError as e:
        logger.error("Spec generation failed: %s", e)
    
    llm_code = await llm.generate_code_from_spec(contract)
    code = extract_content_from_response(llm_code)

    if not code:
        #TODO Change this error 
        raise ValueError("Failed code extraction")
    
    logger.info("Writing code...")
    write_generated_code(code, "experiments/code.py")

    logger.info("Executing code...")
    results = executor.execute_generated_code(code, 10000, 42)
    x, y = results
    dataframe = build_dataframe(x, y)
    save_dataset_csv(dataframe, "experiments/dataset.csv")
    #executor.visualize_data(results)
    
    contract = Contract.from_spec(spec)
    dataset_validator = DatasetValidator(contract)

    df = load_dataset_csv("experiments/dataset.csv")
    validated = dataset_validator.validate_dataset(df)

    if validated:
        logging.info("Dataset is correct according to given schema")

    return

async def main():
    logging.basicConfig(
        level=logging.DEBUG,  # change to DEBUG when needed
        #format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        format="%(levelname)s | %(name)s | %(message)s"
    )

    load_dotenv()
    await run_pipeline()

if __name__ == "__main__":
    asyncio.run(main())