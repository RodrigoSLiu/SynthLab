from src.validation.spec_validator import SpecValidatorError

import logging

logger = logging.getLogger(__name__)


def create_code_generation_prompt():
    return

def add_examples_to_prompt():
    return

def handle_spec_validation_error(e: SpecValidatorError):
    logger.debug(e.details)

    return
