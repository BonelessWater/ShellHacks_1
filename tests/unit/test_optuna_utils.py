import os
import json
import types
from unittest import mock
import sys

import pytest

from backend.ml import optuna_utils


class DummyTrial:
    def __init__(self, number, params, value):
        self.number = number
        self.params = params
        self.value = value


class DummyStudy:
    def __init__(self, name="dummy"):
        self.study_name = name


def test_run_optuna_study_logs_trials(tmp_path, monkeypatch):
    # Prepare a fake optuna that will call the callback with two dummy trials
    def fake_create_study(direction, storage=None, study_name=None):
        study = DummyStudy(name=study_name or "study")

        def optimize(obj, n_trials=None, callbacks=None):
            # Simulate two trials
            t1 = DummyTrial(0, {"x": 0.1}, 0.2)
            t2 = DummyTrial(1, {"x": 0.6}, 0.1)
            if callbacks:
                for cb in callbacks:
                    cb(study, t1)
                    cb(study, t2)

        study.optimize = optimize
        return study

    fake_optuna = types.SimpleNamespace(create_study=fake_create_study)

    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    # Ensure opt-in env var is set
    os.environ["RUN_OPTUNA"] = "1"

    log_path = str(tmp_path / "trials.jsonl")

    def dummy_objective(trial):
        return 1.0

    # Run the helper — it should use our fake optuna and write two JSON lines
    res = optuna_utils.run_optuna_study(dummy_objective, n_trials=2, trial_log_path=log_path, require_opt_in=True)
    assert isinstance(res, dict)

    # Validate the log file
    with open(log_path, "r") as fh:
        lines = [json.loads(l) for l in fh.readlines()]

    assert len(lines) == 2
    assert "trial_number" in lines[0]
    assert "params" in lines[0]

