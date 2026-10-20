from src.errors import ParserError

import logging

logger = logging.getLogger(__name__)


def create_code_generation_prompt():
    return

def add_examples_to_prompt():
    return

def handle_parser_error(e: ParserError):
    logger.debug(e.details)

    return
