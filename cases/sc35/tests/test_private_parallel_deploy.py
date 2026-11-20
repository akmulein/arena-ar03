import pytest

from cases.sc35.tests.test_deploy import DEPLOY, compose_config


@pytest.mark.parametrize("override", [None, "private:updated"])
def test_private_coordinator_updates_without_changing_other_services_or_runners(tmp_path, override):
    overrides = {"ARENA_PRIVATE_EVALUATOR_IMAGE": override} if override is not None else {}
    config = compose_config(tmp_path, DEPLOY / "compose.yaml", **overrides)
    private = config["services"]["private-evaluator"]
    assert private["image"] == (override or "evaluator:synthetic")
    assert int(private["mem_limit"]) == 24 * 1024**3
    for name in ("train-evaluator", "public-evaluator", "model-gateway", "researcher-gateway"):
        service = config["services"][name]
        assert service["image"] == "evaluator:synthetic"
        assert int(service["mem_limit"]) == 4 * 1024**3
    settings = private["environment"]
    assert settings["ARENA_EXECUTION_SLOTS"] == "4"
    assert settings["ARENA_RUNNER_MEMORY"] == "32g"
    assert settings["ARENA_RUNNER_CPUS"] == "8"
