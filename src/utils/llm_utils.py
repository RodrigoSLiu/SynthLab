from src.validation.validator import ValidationError
import os


def create_code_generation_prompt():
    return

def add_examples_to_prompt():
    return

def handle_validation_error(e: ValidationError):
    if os.getenv("DEBUG"):
        print(e.details)

    return