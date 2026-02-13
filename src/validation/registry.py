from checks.distribution import run_ks_test

VALIDATION_REGISTRY = {
    "ks": run_ks_test,
}