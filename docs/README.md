# Synthetic Dataset Generator

A declarative, contract-driven system for generating synthetic datasets using LLM-assisted specification and deterministic execution.

This project focuses on building **reproducible, validated data-generating processes**, not arbitrary AI-written scripts.

---

# Core Philosophy

The system separates:

- **Specification** (LLM-assisted, untrusted)
- **Validation** (schema-enforced, authoritative)
- **Execution** (deterministic primitives only)
- **Statistical verification** (post-generation validation)

The LLM is used to remove friction when writing dataset specifications —  
it does **not** generate executable code.

All dataset generation is performed by a controlled execution engine using an allowed registry of primitives.

---

# Architecture

User Intent  
→ LLM  
→ YAML Specification (untrusted)  
→ JSON Schema Validation  
→ Contract (trusted representation)  
→ Deterministic Executor (allowed primitives only)  
→ Dataset  
→ Dataset Validator  
→ Report

---

# What This Project Is

A system that:

- Translates natural language into structured dataset specifications
- Enforces strict schema validation before execution
- Generates synthetic datasets deterministically using seeded RNG
- Validates statistical properties against declared assumptions
- Enables controlled ML robustness testing under defined distribution shifts

---

# Specification System

The dataset structure and statistical assumptions are defined via YAML.

Example:

```yaml
variables:
    x:
        type: continuous
    y:
        type: continuous

assumptions:
    marginals:
        - variable: x
          generator:
              name: uniform
              params:
                  low: 0.0
                  high: 1.0

    relationships:
        - target: y
          inputs: [x]
          function:
              name: linear
              params:
                  slope: 2.0
                  intercept: 0.0
          noise:
              name: normal
              params:
                  mean: 0.0
                  std: 1.0

validation:
    - type: distribution
      variable: x
      test: ks
      alpha: 0.05

    - type: slope
      target: y
      inputs: [x]
      expected: 2.0
      tolerance: 0.2
```

You should use:
• Ruff (lint + style)
• MyPy (types)
• Optional: Black or Ruff formatter
