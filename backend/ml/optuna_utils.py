"""Optional Optuna helpers (run only when Optuna is installed).

These helpers provide a small wrapper to run an Optuna study for model
hyperparameter optimization. If Optuna isn't installed the functions are
no-ops or raise a helpful error.
"""
from typing import Any, Callable, Dict, Optional


def run_optuna_study(objective: Callable, n_trials: int = 20, storage: Optional[str] = None, study_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        import optuna  # type: ignore

        study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage)
        study.optimize(objective, n_trials=n_trials)
        return {"best_params": study.best_params, "best_value": study.best_value}
    except Exception as e:
        raise RuntimeError("Optuna not available or study failed: " + str(e))
