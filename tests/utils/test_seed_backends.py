"""Cover the optional Torch/JAX seeding branches by injecting fake modules.

Torch and JAX are optional extras absent from the base CI env, so the real
branches never execute there. We inject minimal fakes to exercise the seeding
logic itself (not the libraries).
"""

import sys
import types

from drrl.utils import seed as seed_mod


def _fake_torch() -> types.ModuleType:
    mod = types.ModuleType("torch")
    calls: dict[str, object] = {}

    def manual_seed(s: int) -> None:
        calls["manual_seed"] = s

    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda s: calls.__setitem__("cuda_seed", s),
    )
    cudnn = types.SimpleNamespace(deterministic=False, benchmark=True)

    mod.manual_seed = manual_seed  # type: ignore[attr-defined]
    mod.cuda = cuda  # type: ignore[attr-defined]
    mod.backends = types.SimpleNamespace(cudnn=cudnn)  # type: ignore[attr-defined]
    mod.use_deterministic_algorithms = (  # type: ignore[attr-defined]
        lambda v, warn_only=False: calls.__setitem__("det", v)
    )
    mod._calls = calls  # type: ignore[attr-defined]
    return mod


def _fake_jax() -> types.ModuleType:
    mod = types.ModuleType("jax")
    random_ns = types.SimpleNamespace(PRNGKey=lambda s: ("key", s))
    mod.random = random_ns  # type: ignore[attr-defined]
    return mod


def test_seed_torch_branch(monkeypatch):
    fake = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    seeded: list[str] = []
    det: list[str] = []
    seed_mod._seed_torch(123, deterministic=True, seeded=seeded, det=det)
    assert "torch" in seeded
    assert "torch" in det
    assert fake._calls["manual_seed"] == 123
    assert fake._calls["det"] is True
    assert fake.backends.cudnn.deterministic is True


def test_seed_torch_non_deterministic(monkeypatch):
    fake = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    seeded: list[str] = []
    det: list[str] = []
    seed_mod._seed_torch(7, deterministic=False, seeded=seeded, det=det)
    assert seeded == ["torch"]
    assert det == []  # not flagged deterministic


def test_seed_jax_branch(monkeypatch):
    fake = _fake_jax()
    monkeypatch.setitem(sys.modules, "jax", fake)
    seeded: list[str] = []
    seed_mod._seed_jax(5, seeded=seeded)
    assert seeded == ["jax"]


def test_set_seed_includes_injected_backends(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "jax", _fake_jax())
    state = seed_mod.set_seed(0)
    assert "torch" in state.seeded
    assert "jax" in state.seeded
    assert "torch" in state.deterministic_backends
