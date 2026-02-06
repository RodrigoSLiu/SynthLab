import json 


def create_spec_prompt(schema, intent) -> str:
    schema_text = json.dumps(schema, indent=2)

    return f"""
                You are a specification generator.

                Your task is to generate a YAML specification that STRICTLY conforms to the JSON Schema provided below.

                IMPORTANT RULES:
                - Output ONLY valid YAML.
                - Do NOT include explanations, comments, or markdown.
                - Do NOT invent fields not defined in the schema.
                - Do NOT omit required fields.
                - All values must respect the types and constraints defined in the schema.
                ALL VARIABLES INVOLVED IN CONTINUOUS DISTRIBUTIONS OR ADDITIVE NOISE MUST HAVE TYPE float.
                DO NOT USE int FOR ANY VARIABLE IN THIS SPEC.
                
                JSON SCHEMA (AUTHORITATIVE):
                {schema_text}

                USER INTENT:
                {intent}

                Generate the YAML specification now.
                """
    
def create_code_prompt(contract) -> str:
    return f"""
                You are generating Python code for a synthetic data generator.

                You MUST follow these rules strictly:
                - Output ONLY valid Python code.
                - Do NOT include explanations, comments, or markdown.
                - Do NOT import any libraries except numpy.
                - Do NOT perform validation, logging, printing, or file I/O.
                - Define exactly ONE function with the specified signature.
                - The function must be deterministic given the seed.
                - Do NOT introduce any statistical assumptions that are not explicitly specified.
                - If you include anything other than valid Python code, your output will be rejected.
                - Do not wrap the code in markdown.
                - Do not include comments or explanations.

                ### Function contract
                Function name: relationship_generation

                Inputs:
                - num_samples: int (number of rows to generate)
                - seed: int (random seed)

                Outputs:
                - Return one numpy array per variable, in the same order as defined below.
                - Each array must have length num_samples.
                - Do NOT return a single stacked array, matrix, dict, or object.
                - Each variable must be returned as a separate numpy array.

                ### Dataset schema
                Variables (name → type):
                {contract.variables}

                ### Statistical assumptions

                Marginal distributions (apply exactly as specified):
                {contract.marginals}

                Relationships (apply exactly as specified, including any noise models if present):
                {contract.relationships}

                ### Randomness requirements
                - Use numpy.random.default_rng(seed) as the ONLY source of randomness.
                - All randomness must originate from this RNG.

                ### Behavioral constraints
                - Generate data only.
                - Use vectorized operations (no Python loops if possible).
                - Do NOT infer missing assumptions.
                - Do NOT modify or omit any specified distributions or relationships.
                - If a relationship includes a noise component, apply it exactly as specified.
                - If no noise is specified, the relationship must be deterministic.

                Generate the Python function now.
                """