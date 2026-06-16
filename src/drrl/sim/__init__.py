"""ODE integration and forward sensitivities (backend-swappable).

The diffrax backend is imported lazily (it pulls in JAX, an optional ``sim``
extra) so that importing :mod:`drrl.sim` does not hard-depend on JAX.
"""

from drrl.sim.backend import Backend, SimConfig, compile_model
from drrl.sim.fitting import fit_params, set_natural_params
from drrl.sim.result import SimulationResult


def get_backend(name: str, config: SimConfig | None = None) -> Backend:
    """Return a simulator backend by name.

    Args:
        name: ``"diffrax"`` (JAX, autodiff sensitivities) or ``"scipy"``
            (independent cross-check, finite-difference sensitivities).
        config: Integration controls.

    Returns:
        A :class:`Backend`.

    Raises:
        ValueError: For an unknown backend name.
    """
    if name == "diffrax":
        from drrl.sim.diffrax_backend import DiffraxBackend

        return DiffraxBackend(config)
    if name == "scipy":
        from drrl.sim.scipy_backend import ScipyBackend

        return ScipyBackend(config)
    raise ValueError(f"unknown sim backend {name!r}")


__all__ = [
    "Backend",
    "SimConfig",
    "SimulationResult",
    "compile_model",
    "fit_params",
    "get_backend",
    "set_natural_params",
]
