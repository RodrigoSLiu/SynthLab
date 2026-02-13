# Synthetic Dataset Generator

This project explores how to generate synthetic datasets using **LLM-assisted code generation**, combined with **explicit statistical validation** and **human-in-the-loop review**.

The goal is not to blindly generate data, but to **define, execute, and validate data-generating processes** in a controlled and reproducible way.

---

## Motivation & References

This project is inspired by work on LLM-based code generation and local LLM usage:

- Local LLM benchmarks and setup:  
  https://www.nunariq.com/knowledgebase/building-an-ai-python-code-generator-with-local-llms/

- LLM code generation overview:  
  https://mskadu.medium.com/generating-code-with-llms-a-developers-guide-part-1-0c381dc3e57a

- Local LLM used:  
  https://huggingface.co/mistralai/Codestral-22B-v0.1

---

## Project Definition

### What is it?

The Synthetic Dataset Generator (SDG) generates synthetic datasets from **natural language intent**.

An LLM is used to translate user intent into **Python code that defines a data-generating process**.  
The generated code is **reviewed by a human**, executed in a **sandboxed environment**, and the resulting dataset is **validated using statistical tests**.

The system does not trust LLM outputs by default — correctness is enforced through validation.

---

## Core Flow

User intent
   ↓
LLM
   ↓
YAML spec (untrusted)
   ↓
SpecValidator 
   ↓
Contract (trusted)
   ↓
Code generation
   ↓
Data generation
   ↓
DatasetValidator
   ↓
Results / report

---

## MVP Scope

The MVP focuses on proving the core loop works.

Included:

- Conversion of user intent into a predefined YAML specification (contract + validation metadata)
- LLM-generated Python code
- Human-editable Python code
- Generation of tabular data
- Ability to determine which statistical validations to run
- Simple statistical validation with clear pass/fail outcomes

---

## Nice to Have (Post-MVP)

- Iterative mode: if validation fails, regenerate and compare
- Multiple experiments per intent
- Comparison of outputs across runs

---

## Technology Choices

### High Level

| Component           | Purpose                          | Technology                      |
| ------------------- | -------------------------------- | ------------------------------- |
| LLM                 | Translate intent into code       | Codestral-22B-v0.1 (local)      |
| Code language       | Data generation logic            | Python                          |
| Spec format         | Contract & validation metadata   | YAML                            |
| YAML parsing        | Read specs                       | pyyaml                          |
| Execution sandbox   | Safe execution of generated code | Python AST                      |
| Validation engine   | Statistical validation           | scipy.stats + custom validators |
| Experiment tracking | Store results & comparisons      | JSON                            |

---

## Execution Contract (Conceptual)

Each execution produces either a **valid result** or a **failed experiment**.

#### Example

##### Intent:

Generate a dataset with two columns x and y, where:
• x is uniformly sampled from the interval (0, 1)
• y = 2x + noise
• noise follows a zero-mean Gaussian distribution

##### Contract:

Example generated function output:

```json
{
    "code_contract": {
        "language": "python3.10",

        "function": {
            "name": "relationship_generation",
            "description": "Generate a synthetic dataset according to predefined statistical assumptions.",
            "inputs": {
                "num_samples": {
                    "type": "int",
                    "required": true,
                    "constraints": {
                        "min": 1,
                        "max": 1000000
                    }
                },
                "seed": {
                    "type": "int",
                    "required": true,
                    "description": "Seed used to initialize deterministic random number generation."
                }
            },
            "outputs": {
                "x": {
                    "type": "array",
                    "dtype": "float",
                    "length": "num_samples"
                },
                "y": {
                    "type": "array",
                    "dtype": "float",
                    "length": "num_samples"
                }
            }
        },

        "environment": {
            "allowed_imports": ["numpy"],
            "forbidden_imports": [
                "os",
                "sys",
                "subprocess",
                "pathlib",
                "requests",
                "matplotlib",
                "seaborn"
            ]
        },

        "randomness": {
            "deterministic": true,
            "rng_library": "numpy.random",
            "rng_pattern": "numpy.random.default_rng(seed)"
        },

        "execution_constraints": {
            "no_file_io": true,
            "no_network_calls": true,
            "no_printing": true,
            "no_plotting": true,
            "no_global_state": true,
            "no_recursion": true,
            "time_limit_seconds": 2
        },

        "behavioral_rules": {
            "must_generate_data_only": true,
            "must_not_perform_validation": true,
            "must_not_log_or_save_results": true
        }
    }
}
```

```python
import numpy as np

def relationship_generation(num_samples: int, seed: int):
    rng = np.random.default_rng(seed)

    x = rng.uniform(0.0, 1.0, size=num_samples)
    noise = rng.normal(loc=0.0, scale=1.0, size=num_samples)
    y = 2.0 * x + noise

    return x, y
```

#### Validation

```json
{
    "validation_requirements": [
        {
            "check": "schema_check",
            "columns": ["x", "y"],
            "types": {
                "x": "float",
                "y": "float"
            }
        },
        {
            "check": "distribution_test",
            "column": "x",
            "distribution": "uniform",
            "params": {
                "min": 0.0,
                "max": 1.0
            },
            "test": "ks",
            "alpha": 0.05
        },
        {
            "check": "relationship_check",
            "type": "linear_regression",
            "dependent": "y",
            "independent": "x",
            "expected_slope": 2.0,
            "tolerance": 0.2
        },
        {
            "check": "noise_mean_check",
            "expected_mean": 0.0,
            "tolerance": 0.2
        }
    ]
}
```

Success:

```json
{
    "status": "PASSED",
    "metrics": {
        "mean_x": 0.498,
        "std_noise": 0.99,
        "estimated_slope": 2.03
    }
}
```

Failure:

```json
{
    "status": "FAILED",
    "violations": [
        {
            "check": "relationship_check",
            "expected_slope": 2.0,
            "observed_slope": 1.52,
            "tolerance": 0.2
        },
        {
            "check": "distribution_test",
            "column": "x",
            "test": "ks",
            "p_value": 0.01
        }
    ]
}
```
