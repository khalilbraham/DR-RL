"""Tier-A gate: codegen + integration, with failures captured as data.

A model that fails to compile or whose integrator diverges is not an exception
to propagate — it is a report. This keeps the verifier total: every candidate
yields a structured verdict.
"""

from __future__ import annotations

from drrl.sim import SimulationResult
from drrl.sim.backend import Backend
from drrl.spec.model import Design, ModelSpec
from drrl.verifier.report import ExecutionReport


def check_execution(
    spec: ModelSpec, design: Design, backend: Backend
) -> tuple[ExecutionReport, SimulationResult | None]:
    """Compile and integrate ``spec`` under ``design``.

    Args:
        spec: The model to run.
        design: Dosing + sampling design.
        backend: Simulator backend.

    Returns:
        ``(report, result)`` where ``result`` is the :class:`SimulationResult`
        on success, else ``None``. Any exception (codegen, integration) is
        captured in the report.
    """
    try:
        result = backend.simulate(spec, design, with_sensitivities=False)
    except NotImplementedError as exc:
        return ExecutionReport(
            ok=False, integrator_ok=False, message=f"unsupported: {exc}"
        ), None
    except Exception as exc:
        return ExecutionReport(
            ok=False, integrator_ok=False, message=f"{type(exc).__name__}: {exc}"
        ), None

    import numpy as np

    finite = bool(np.all(np.isfinite(result.observed)))
    ok = result.integrator_ok and finite
    msg = (
        "" if ok else ("non-finite predictions" if not finite else "integrator failed")
    )
    return ExecutionReport(
        ok=ok, integrator_ok=result.integrator_ok, message=msg
    ), result
