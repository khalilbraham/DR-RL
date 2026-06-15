"""scipy backend: an independent integrator for cross-checking correctness.

This backend uses a different numerical method (``scipy.integrate.solve_ivp``)
and finite-difference sensitivities, so agreement with the diffrax backend is
genuine cross-validation rather than a tautology.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sympy
from scipy.integrate import solve_ivp

from drrl.sim.backend import (
    CompiledModel,
    SimConfig,
    apply_observation,
    compile_model,
    dose_schedule,
)
from drrl.sim.result import SimulationResult
from drrl.spec.model import Design, ModelSpec

NumpyVF = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _build_numpy_vf(spec: ModelSpec) -> NumpyVF:
    """Lambdify the ODE RHS to a NumPy ``f(y, theta) -> dy``."""
    syms = spec.symbols
    state_syms = [syms[n] for n in spec.state_names]
    param_syms = [syms[n] for n in spec.param_names]
    rhs = spec.rhs_exprs()
    exprs = [rhs[n] for n in spec.state_names]
    func = sympy.lambdify((state_syms, param_syms), exprs, modules="numpy")

    def vf(y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.asarray(func(y, theta), dtype=np.float64).reshape(-1)

    return vf


class ScipyBackend:
    """Forward simulator backed by ``scipy.integrate.solve_ivp``."""

    def __init__(
        self, config: SimConfig | None = None, *, method: str = "LSODA"
    ) -> None:
        """Initialize with integration controls and a scipy method."""
        self.config = config or SimConfig()
        self.method = method

    def _states(
        self,
        vf: NumpyVF,
        compiled: CompiledModel,
        sample_times: tuple[float, ...],
        theta: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        initial, future = dose_schedule(compiled)
        y = np.zeros(compiled.n_states, dtype=np.float64)
        for idx, amt in initial:
            y[idx] += amt
        outs: list[np.ndarray] = []
        ok_all = True
        t_cur = 0.0
        fi = 0

        def integrate(t0: float, t1: float, y0: np.ndarray) -> tuple[np.ndarray, bool]:
            sol = solve_ivp(
                lambda t, yy: vf(yy, theta),
                (t0, t1),
                y0,
                method=self.method,
                rtol=self.config.rtol,
                atol=self.config.atol,
                dense_output=False,
            )
            return sol.y[:, -1], bool(sol.success)

        for ts in sample_times:
            while fi < len(future) and future[fi][0] <= ts:
                dt, idx, amt = future[fi]
                if dt > t_cur:
                    y, ok = integrate(t_cur, dt, y)
                    ok_all = ok_all and ok
                    t_cur = dt
                y = y.copy()
                y[idx] += amt
                fi += 1
            if ts > t_cur:
                y, ok = integrate(t_cur, ts, y)
                ok_all = ok_all and ok
                t_cur = ts
            outs.append(y.copy())
        return np.stack(outs), ok_all

    def simulate(
        self, spec: ModelSpec, design: Design, *, with_sensitivities: bool = False
    ) -> SimulationResult:
        """Integrate ``spec`` under ``design`` (see ``Backend``)."""
        compiled = compile_model(spec, design)
        vf = _build_numpy_vf(spec)
        theta = compiled.theta_natural

        states, ok = self._states(vf, compiled, design.sample_times, theta)
        observed = apply_observation(states, compiled)

        sens: np.ndarray | None = None
        if with_sensitivities:
            sens = self._finite_diff_sensitivities(spec, design, compiled, vf, theta)

        return SimulationResult(
            times=np.asarray(design.sample_times, dtype=np.float64),
            states=states,
            observed=observed,
            sensitivities=sens,
            integrator_ok=ok,
            diagnostics={"backend": "scipy", "method": self.method},
        )

    def _finite_diff_sensitivities(
        self,
        spec: ModelSpec,
        design: Design,
        compiled: CompiledModel,
        vf: NumpyVF,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Central finite-difference ``d observed / d theta``, shape (T, n_params)."""
        n_t = len(design.sample_times)
        n_p = len(theta)
        jac = np.zeros((n_t, n_p), dtype=np.float64)
        for j in range(n_p):
            step = 1e-6 * max(abs(theta[j]), 1.0)
            tp = theta.copy()
            tp[j] += step
            cm_p = _with_theta(compiled, tp)
            sp, _ = self._states(vf, cm_p, design.sample_times, tp)
            op = apply_observation(sp, cm_p)
            tm = theta.copy()
            tm[j] -= step
            cm_m = _with_theta(compiled, tm)
            sm, _ = self._states(vf, cm_m, design.sample_times, tm)
            om = apply_observation(sm, cm_m)
            jac[:, j] = (op - om) / (2 * step)
        return jac


def _with_theta(compiled: CompiledModel, theta: np.ndarray) -> CompiledModel:
    """Copy a compiled model with a perturbed theta (for finite differences)."""
    from dataclasses import replace

    return replace(compiled, theta_natural=theta)
