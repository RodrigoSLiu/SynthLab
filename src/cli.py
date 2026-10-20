from src.llm.ollama_client import OllamaClient
from src.domain.contract import Contract
from src.domain.parser import Parser
from src.validation.dataset_validation import DatasetValidator
from src.errors import ParserError
from src.utils.utils import Utils
from src.execution.executor import Executor

from dotenv import load_dotenv
from typing import Optional, Dict, Any
import json
import os
import asyncio
import logging
import argparse


logger = logging.getLogger(__name__)

async def get_spec_with_retry(
    llm,
    spec_parser: Parser,
    schema: Dict[str, Any],
    intent: str,
    retry: int = 1,
) -> Dict[str, Any]:

    yaml_text: str = ""
    last_error: Optional[ParserError] = None

    for i in range(retry):
        logger.info("Trying %d/%d...", i + 1, retry)

        if last_error is not None:
            logger.warning("Spec parsing failed, re-running...")
            llm_spec = await llm.regenerate_spec_from_error(last_error, yaml_text)
        else:
            llm_spec = await llm.generate_spec_from_intent(schema, intent)
            pass

        try:
            logger.info("#### EXTRACTED YAML #####")
            yaml_text = Utils.extract_content_from_response(llm_spec)
            logger.info(yaml_text)

            logger.info("#### SPEC PARSING #####")
            spec = spec_parser.parse_yaml(yaml_text)

            return spec

        except ParserError as e:
            last_error = e
            logger.error("Parser raised error: %s", e)

    # If we reach here, all retries failed
    raise ParserError(
        error_type="SPEC_GENERATION_FAILED",
        details={
            "message": f"Failed after {retry} attempts",
            "last_error_type": last_error.error_type if last_error else None,
            "last_error_details": last_error.details if last_error else None,
            "last_yaml_attempt": yaml_text,
        },
    )


async def run_pipeline():
    schema = Utils.load_schema("specification.schema.json")
    spec_parser = Parser(schema)

    spec_model_name = os.getenv("SPEC_LLM_MODEL")
    code_model_name = os.getenv("CODE_LLM_MODEL")
    llm = OllamaClient(spec_model_name, code_model_name)

    executor = Executor()

    intent = (
        '''I have a regression model that assumes additive feature effects.
        I want to test whether it fails when the true data-generating process contains multiplicative feature interactions.

        Generate a dataset where:
        • x1 follows a normal distribution with mean 0 and standard deviation 1.
        • x2 follows a uniform distribution between -1 and 1.
        • y is defined as y = 1.2*x1 + 0.8*x2 + 2.0*(x1*x2).
        • The relationship must be fully deterministic.
        • Validate that both marginals match their specified distributions.
        • Validate that the interaction coefficient matches 2.0 within tolerance.
        '''
    )
    
    if args.filename:
        with open(f"specs/{args.filename}", "r", encoding="utf-8") as f:
            yaml_text = f.read()

        spec = spec_parser.parse_yaml(yaml_text)
    else:
        spec = await get_spec_with_retry(llm, spec_parser, schema, intent, retry=3)

    try:
        logger.info("Writing yaml...")
        Utils.write_generated_yaml(spec, "specs/test.yaml")
        contract = Contract.from_spec(spec)
    except ValueError as e:
        logger.error("Spec generation failed: %s", e)

    #logger.info("Writing code...")
    #Utils.write_generated_code(code, "experiments/code.py")

    #logger.info("Executing code...")
    dataframe = executor.execute_spec(contract, 10000, 42)
    logger.info(len(dataframe))
    Utils.save_dataset_csv(dataframe, "experiments/dataset.csv")

    
    contract = Contract.from_spec(spec)
    dataset_validator = DatasetValidator(contract)

    df = Utils.load_dataset_csv("experiments/dataset.csv")
    executor.visualize_data(df)
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
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    load_dotenv()
    await run_pipeline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SynthLab')
    parser.add_argument('-f', '--filename')
    args = parser.parse_args()
    asyncio.run(main())