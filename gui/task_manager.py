"""Background task management for long-running AutoEIS computations.

Single-process design: suitable for one user at a time (local / lab use) or
low-traffic public deployments.  State is keyed by task_id so multiple
simultaneous runs don't collide.

Public-use guards:
  - MAX_CONCURRENT_TASKS: cap on simultaneous running analyses (env var).
  - Tasks are automatically expired and removed after TASK_TTL_S seconds so
    the in-process dict doesn't grow unboundedly.
"""

import os
import threading
import time
import uuid
from typing import Any

_tasks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_TASKS", "3"))
TASK_TTL_S = int(os.environ.get("TASK_TTL_S", str(60 * 60)))  # 1 hour default
TASK_TIMEOUT_S = int(os.environ.get("TASK_TIMEOUT_S", str(30 * 60)))  # 30 min hard kill


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _active_count() -> int:
    with _lock:
        return sum(1 for t in _tasks.values() if t["status"] == "running")


def _expire_old_tasks():
    """Remove tasks older than TASK_TTL_S. Called on each create_task()."""
    now = time.time()
    with _lock:
        stale = [tid for tid, t in _tasks.items() if now - t["created_at"] > TASK_TTL_S]
        for tid in stale:
            del _tasks[tid]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_task() -> str | None:
    """Return a new task ID, or None if the server is at capacity."""
    _expire_old_tasks()
    if _active_count() >= MAX_CONCURRENT:
        return None
    tid = str(uuid.uuid4())
    with _lock:
        _tasks[tid] = {
            "status": "pending",   # pending | running | done | error
            "progress": 0,
            "stage": "",
            "circuits": None,      # list[dict] for display in fitting page
            "results": None,       # list[serialised InferenceResult]
            "error": "",
            "created_at": time.time(),
        }
    return tid


def create_error_task(message: str) -> str:
    """Create a task pre-filled with an error message (used for capacity rejection)."""
    tid = str(uuid.uuid4())
    with _lock:
        _tasks[tid] = {
            "status": "error",
            "progress": 0,
            "stage": "",
            "circuits": None,
            "results": None,
            "error": message,
            "created_at": time.time(),
        }
    return tid


def _update(tid: str, **kw):
    with _lock:
        if tid in _tasks:
            _tasks[tid].update(kw)


def get_task(tid: str) -> dict:
    with _lock:
        return dict(_tasks.get(tid, {}))


def delete_task(tid: str):
    with _lock:
        _tasks.pop(tid, None)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def run_analysis(tid: str, freq, Z, mode_config: dict):
    """Launch the AutoEIS pipeline in a daemon thread."""

    def _worker():
        import numpy as np
        import autoeis as ae

        _update(tid, status="running", progress=2, stage="Initialising…")
        deadline = time.time() + TASK_TIMEOUT_S

        def _check_timeout():
            if time.time() > deadline:
                raise TimeoutError(
                    f"Analysis exceeded the {TASK_TIMEOUT_S // 60}-minute time limit."
                )

        try:
            mode = mode_config.get("mode", "auto")
            params = mode_config.get("params", {})

            # ----------------------------------------------------------------
            # Stage 1: generate / select candidate circuits
            # ----------------------------------------------------------------
            if mode == "quick":
                _update(tid, progress=10, stage="Using pre-defined circuits…")
                circuit_strings = mode_config.get("circuits", [])
                if not circuit_strings:
                    _update(tid, status="error", error="No circuits selected. Go back to Mode and select at least one.")
                    return
                circuits_for_bi = circuit_strings
                circuits_records = [
                    {"circuitstring": c, "Parameters": dict.fromkeys(ae.parser.get_parameter_labels(c))}
                    for c in circuit_strings
                ]

            elif mode == "expert":
                _update(tid, progress=10, stage="Using expert circuit…")
                circuit_str = (mode_config.get("circuit_str") or "").strip()
                if not circuit_str:
                    _update(tid, status="error", error="No circuit string provided. Go back to Mode and enter a circuit.")
                    return
                circuits_for_bi = [circuit_str]
                circuits_records = [
                    {"circuitstring": circuit_str, "Parameters": dict.fromkeys(ae.parser.get_parameter_labels(circuit_str))}
                ]

            else:  # auto
                _check_timeout()
                _update(tid, progress=5, stage="Generating candidate circuits (evolutionary search)…")
                circuits_unfiltered = ae.core.generate_equivalent_circuits(
                    freq, Z,
                    iters=int(params.get("iters", 100)),
                    complexity=int(params.get("complexity", 12)),
                    population_size=int(params.get("population_size", 100)),
                    generations=int(params.get("generations", 30)),
                    tol=float(params.get("tol", 1e-2)),
                    terminals=params.get("terminals", "RLP"),
                    parallel=True,
                )
                if circuits_unfiltered is None or len(circuits_unfiltered) == 0:
                    _update(tid, status="error", error="Circuit generation returned no candidates. Try relaxing tol or increasing iters.")
                    return
                n_unfiltered = len(circuits_unfiltered)
                _check_timeout()
                _update(tid, progress=45, stage=f"Filtering circuits ({n_unfiltered} candidates)…")
                circuits_filtered = ae.core.filter_implausible_circuits(circuits_unfiltered)
                if circuits_filtered is None or len(circuits_filtered) == 0:
                    _update(tid, status="error", error="All generated circuits were filtered as implausible. Try different hyperparameters.")
                    return
                circuits_for_bi = circuits_filtered
                circuits_records = circuits_filtered.to_dict("records")

            n_circuits = len(circuits_for_bi)
            _update(tid, progress=50, stage=f"{n_circuits} circuit(s) ready for inference", circuits=circuits_records)

            if n_circuits == 0:
                _update(tid, status="error", error="No valid circuits found.")
                return

            # ----------------------------------------------------------------
            # Stage 2: Bayesian inference
            # ----------------------------------------------------------------
            _check_timeout()
            _update(tid, progress=55, stage="Running Bayesian inference (this may take several minutes)…")
            results = ae.core.perform_bayesian_inference(
                circuits_for_bi,
                freq,
                Z,
                num_warmup=int(params.get("num_warmup", 2500)),
                num_samples=int(params.get("num_samples", 1000)),
                num_chains=1,
                refine_p0=True,
                parallel=False,
                progress_bar=False,
            )

            if not isinstance(results, list):
                results = [results]

            # ----------------------------------------------------------------
            # Serialise results for JSON storage in dcc.Store
            # ----------------------------------------------------------------
            serialised = []
            for r in results:
                if r is None:
                    serialised.append(None)
                    continue
                if r.converged:
                    samples = {k: np.array(v).tolist() for k, v in r.samples.items()}
                    try:
                        num_div = int(r.num_divergences)
                    except Exception:
                        num_div = 0
                else:
                    samples = {}
                    num_div = 0
                serialised.append(
                    {
                        "circuit": r.circuit,
                        "converged": bool(r.converged),
                        "variables": list(r.variables),
                        "samples": samples,
                        "num_divergences": num_div,
                        "freq": np.array(r.freq).tolist(),
                        "z_real": np.array(r.Z.real).tolist(),
                        "z_imag": np.array(r.Z.imag).tolist(),
                    }
                )

            _update(
                tid,
                status="done",
                progress=100,
                stage="Analysis complete!",
                results=serialised,
            )

        except Exception as exc:
            import traceback
            _update(tid, status="error", error=f"{exc}\n\n{traceback.format_exc()}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
