"""diffrax/JAX backend: integration with autodiff forward sensitivities.

PK/PD systems need float64; we enable it on import. Forward sensitivities use
diffrax's ``ForwardMode`` adjoint (the default adjoint is reverse-mode only and
cannot be driven by ``jax.jacfwd``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax
import jax
import numpy as np
import sympy

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]  # jax setter

import jax.numpy as jnp  # noqa: E402  (must follow x64 enable)

from drrl.sim.backend import (  # noqa: E402
    CompiledModel,
    SimConfig,
    compile_model,
    dose_schedule,
)
from drrl.sim.result import SimulationResult  # noqa: E402
from drrl.spec.model import Design, ModelSpec  # noqa: E402

JaxVectorField = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _build_vector_field(spec: ModelSpec) -> JaxVectorField:
    """Lambdify the ODE RHS to a JAX-traceable ``f(y, theta) -> dy``."""
    syms = spec.symbols
    state_syms = [syms[n] for n in spec.state_names]
    param_syms = [syms[n] for n in spec.param_names]
    rhs = spec.rhs_exprs()
    exprs = [rhs[n] for n in spec.state_names]
    func = sympy.lambdify((state_syms, param_syms), exprs, modules=[jnp])

    def vf(y: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(func(y, theta)).reshape(-1)

    return vf


class DiffraxBackend:
    """Forward simulator backed by diffrax (JAX)."""

    def __init__(self, config: SimConfig | None = None) -> None:
        """Initialize with integration controls (defaults if ``None``)."""
        self.config = config or SimConfig()

    def _solver(self) -> Any:
        return getattr(diffrax, self.config.solver)()

    def _states_fn(
        self,
        vf: JaxVectorField,
        compiled: CompiledModel,
        sample_times: tuple[float, ...],
    ) -> Callable[[jnp.ndarray, Any], tuple[jnp.ndarray, bool]]:
        """Build ``theta, adjoint -> (states[T, n_states], integrator_ok)``.

        The full sample grid is integrated with a *single* ``SaveAt(ts=...)``
        solve and times are passed as JAX arrays, not Python floats. Passing
        times as compile-time float constants would trigger one XLA compilation
        per sample time, which is both slow and, on some platforms, exhausts the
        XLA compiler. Phase 1 supports a single IV bolus at ``t == 0``; t>0
        boluses (handled by the scipy reference backend) are deferred.
        """
        initial, future = dose_schedule(compiled)
        if future:
            raise NotImplementedError(
                "diffrax backend supports a single t=0 bolus in Phase 1; "
                "use the scipy backend for t>0 dosing"
            )
        term = diffrax.ODETerm(lambda t, y, a: vf(y, a))
        solver = self._solver()
        controller = diffrax.PIDController(rtol=self.config.rtol, atol=self.config.atol)
        ts_arr = jnp.asarray(sample_times)
        t0 = jnp.asarray(min(0.0, float(sample_times[0])))

        def run(theta: jnp.ndarray, adj: Any) -> tuple[jnp.ndarray, bool]:
            y0 = jnp.zeros(compiled.n_states)
            for idx, amt in initial:
                y0 = y0.at[idx].add(amt)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=t0,
                t1=ts_arr[-1],
                dt0=None,
                y0=y0,
                args=theta,
                stepsize_controller=controller,
                max_steps=self.config.max_steps,
                saveat=diffrax.SaveAt(ts=ts_arr),
                adjoint=adj,
                throw=False,
            )
            ok = bool(sol.result == diffrax.RESULTS.successful)
            return sol.ys, ok

        return run

    @staticmethod
    def _observe(
        states: jnp.ndarray, compiled: CompiledModel, theta: jnp.ndarray
    ) -> jnp.ndarray:
        y = states[:, compiled.obs_state_index]
        if compiled.divide_by_index is not None:
            y = y / theta[compiled.divide_by_index]
        if compiled.log_transform:
            y = jnp.log(y)
        return y

    def simulate(
        self, spec: ModelSpec, design: Design, *, with_sensitivities: bool = False
    ) -> SimulationResult:
        """Integrate ``spec`` under ``design`` (see ``Backend``)."""
        compiled = compile_model(spec, design)
        vf = _build_vector_field(spec)
        run = self._states_fn(vf, compiled, design.sample_times)
        theta = jnp.asarray(compiled.theta_natural)

        # Value pass (reverse-mode default adjoint; cheap, gives integrator status).
        states_j, ok = run(theta, diffrax.RecursiveCheckpointAdjoint())
        states = np.asarray(states_j, dtype=np.float64)
        observed = np.asarray(
            self._observe(states_j, compiled, theta), dtype=np.float64
        )

        sens: np.ndarray | None = None
        if with_sensitivities:

            def predict(th: jnp.ndarray) -> jnp.ndarray:
                st, _ = run(th, diffrax.ForwardMode())
                return self._observe(st, compiled, th)

            jac = jax.jacfwd(predict)(theta)
            sens = np.asarray(jac, dtype=np.float64)

        return SimulationResult(
            times=np.asarray(design.sample_times, dtype=np.float64),
            states=states,
            observed=observed,
            sensitivities=sens,
            integrator_ok=ok,
            diagnostics={"backend": "diffrax", "solver": self.config.solver},
        )
