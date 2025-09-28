"""Optuna helper with optional opt-in trial logging.

This module is intentionally light-weight. It will raise a clear error
if Optuna is not installed. It also provides optional trial logging
to a JSONL file and enforces an opt-in via env var or explicit flag to
prevent accidental long-running runs in CI.
"""
from typing import Any, Callable, Dict, Optional
import os
import json
from datetime import datetime


def _ensure_optuna():
    try:
        import optuna  # type: ignore

        return optuna
    except Exception as e:  # pragma: no cover - optuna may not be present
        raise ImportError("Optuna is not installed. Install optuna to run hyperparameter tuning.") from e


def run_optuna_study(objective: Callable, n_trials: int = 20, storage: Optional[str] = None, study_name: Optional[str] = None,
                     *, trial_log_path: Optional[str] = None, require_opt_in: bool = True, opt_in_env: str = "RUN_OPTUNA") -> Dict[str, Any]:
    """Run an Optuna study with optional trial logging.

    Parameters:
    - objective: function(trial) -> float
    - n_trials: number of trials
    - storage, study_name: passed to optuna.create_study
    - trial_log_path: if provided, append JSON lines with trial metadata
    - require_opt_in: if True, will only run if env[opt_in_env]=="1"
    - opt_in_env: environment variable name to check for opt-in
    """
    if require_opt_in and os.environ.get(opt_in_env, "0") != "1":
        raise RuntimeError(f"Optuna runs are disabled by default. Set {opt_in_env}=1 or call with require_opt_in=False to proceed.")

    optuna = _ensure_optuna()

    study = optuna.create_study(direction="minimize", storage=storage, study_name=study_name)

    # Optional trial logging: write a JSONL line per completed trial
    if trial_log_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(trial_log_path) or ".", exist_ok=True)

        def _callback(study, trial):
            line = {
                "study_name": study.study_name,
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "datetime": datetime.utcnow().isoformat() + "Z",
            }
            with open(trial_log_path, "a") as fh:
                fh.write(json.dumps(line) + "\n")

        study.optimize(objective, n_trials=n_trials, callbacks=[_callback])
    else:
        study.optimize(objective, n_trials=n_trials)

    return {"best_params": getattr(study, "best_params", None), "best_value": getattr(study, "best_value", None)}
