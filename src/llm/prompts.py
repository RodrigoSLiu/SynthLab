from src.domain.contract import Contract

import json 


def create_spec_prompt(schema:str, intent: str) -> str:
    schema_text = json.dumps(schema, indent=2)

    return f"""
            You are a strict YAML specification compiler.

            Your task is to translate USER INTENT into a YAML specification that strictly conforms to the provided JSON Schema.

            The JSON Schema is the sole authority. You must obey it exactly.

            ---------------------------------------
            HARD OUTPUT CONSTRAINTS
            ---------------------------------------

            - Output ONLY valid YAML.
            - Do NOT include explanations.
            - Do NOT include markdown.
            - Do NOT include comments.
            - Do NOT wrap output in backticks.
            - Do NOT include text before or after the YAML.
            - Use block-style YAML only.
            - Indentation MUST be exactly 2 spaces per level.
            - Lists MUST use "-" on new lines.
            - All field names MUST exactly match the schema.
            - Do NOT invent fields.
            - Do NOT omit required fields.
            - All numeric values MUST be valid numbers.

            ---------------------------------------
            PRIMITIVE CATEGORIES (STRICT ONTOLOGY)
            ---------------------------------------

            Marginal generators (assumptions.marginals.generator.name) MUST be one of:

            - uniform
            - normal
            - exponential
            - binomial
            - beta
            - gamma
            - lognormal

            Relationship functions (relationships.function.name) MUST be one of:

            - linear
            - polynomial
            - logistic
            - piecewise_linear

            Noise generators (relationships.noise.name) MUST be:

            - normal

            The following are FORBIDDEN everywhere:
            - expression
            - lambda
            - parametric
            - function strings
            - inline math expressions

            ---------------------------------------
            SECTION RESPONSIBILITIES (STRICT)
            ---------------------------------------

            1. variables
            - Only declares variable names and type.
            - MUST NOT contain generator definitions.
            - MUST NOT contain distribution parameters.
            - MUST NOT contain function definitions.
            - MUST NOT contain params.

            2. assumptions
            - Defines ONLY marginal distributions.
            - Only independent (exogenous) variables may appear here.
            - A variable that appears as a relationship target MUST NOT appear in marginals.
            - Distribution parameters MUST appear ONLY under:
                assumptions -> marginals -> generator -> params

            3. relationships
            - Defines deterministic mappings between variables.
            - Function parameters MUST appear ONLY under:
                relationships -> function -> params
            - Target MUST exist in variables.
            - Inputs MUST exist in variables.
            - If no noise is described in intent, DO NOT include a noise field.

            4. validation
            - Must correspond to existing variables or relationships.
            - Include only fields required by the validation type.
            - For polynomial validation:
                - expected MUST be an array of numbers.
                - Do NOT wrap expected inside another object.
            - For slope validation:
                - expected MUST be a number.
            - Do NOT include irrelevant fields.

            ---------------------------------------
            SEMANTIC CONSISTENCY RULES
            ---------------------------------------

            - Every referenced variable MUST be declared.
            - No variable may appear in both marginals and as a relationship target.
            - Marginal generator names MUST be valid distribution names.
            - Relationship function names MUST be valid function names.
            - Parameter names MUST match expected primitive conventions:
                normal distribution uses:
                    mean
                    std
                (NOT std_dev)
            - If a numeric value is an integer conceptually, still represent it as a float (e.g., 2.0 instead of 2).
            - All numbers in params, expected values, and tolerances MUST be floats.

            ---------------------------------------
            MANDATORY INTERNAL VERIFICATION
            ---------------------------------------

            Before producing output, internally verify:

            - No target variable appears in marginals.
            - No forbidden primitive names appear.
            - No distribution parameters appear outside assumptions.
            - No function parameters appear outside relationships.
            - All required fields are present.
            - No duplicated keys.
            - Validation structure matches schema.

            If any violation exists, correct it before producing final YAML.

            ---------------------------------------
            JSON SCHEMA (AUTHORITATIVE)
            ---------------------------------------

            {schema_text}

            ---------------------------------------
            USER INTENT
            ---------------------------------------

            {intent}

            Generate the YAML specification now.   
            """

def create_yaml_error_prompt(error, invalid_yaml: str) -> str:
    return f"""
                You previously generated a YAML specification, but it is INVALID YAML and failed to parse.

                This is a SYNTAX ERROR ONLY.
                The content and meaning of the YAML are correct — only formatting is wrong.

                ### ❌ Error
                YAML_PARSE_ERROR:
                {error}

                ### ❌ Invalid YAML (DO NOT REPEAT AS-IS)
                {invalid_yaml}

                ---

                ### Your task

                Fix the YAML so that it is:
                - Valid YAML
                - Parsable by yaml.safe_load
                - Semantically identical to the original

                ### STRICT RULES (IMPORTANT)

                - Output ONLY valid YAML
                - DO NOT include explanations, comments, or markdown
                - DO NOT wrap the output in ``` or any formatting
                - DO NOT change field names, values, or structure
                - DO NOT add or remove keys
                - DO NOT infer or introduce new assumptions
                - Fix ONLY indentation, tokens, quoting, or formatting errors

                If you include anything other than valid YAML, the output will be rejected.

                Return the corrected YAML now.
            """
    
def create_code_prompt(contract: Contract) -> str:
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