"""Inference-time best-first edit search guided by the process reward model.

From the initial model, explore structural edits, scoring each candidate by the
PRM's verifier-derived potential (a partial-observability-safe proxy), and commit
the best-scoring structure found. This is multi-turn *repair*: it can fix a wrong
starting structure that a one-shot commit cannot. The terminal reward (the true
distinguishability-relative reward) is then obtained by replaying the chosen edit
sequence through the environment.
"""

from __future__ import annotations

from dataclasses import dataclass

from drrl.env.actions import Action, Commit, EditAction
from drrl.env.environment import ModelEditEnv
from drrl.env.state import ModelState, apply_edit, valid_structural_edits
from drrl.rl.prm import ProcessRewardModel


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a best-first edit search.

    Attributes:
        actions: The edit sequence (ending in Commit) the search selected.
        committed_structure: Structure committed.
        best_potential: PRM potential of the committed structure.
        terminal_reward: True terminal reward from replaying ``actions``.
    """

    actions: tuple[Action, ...]
    committed_structure: str
    best_potential: float
    terminal_reward: float


def _path_to(state: ModelState, target_structure: str) -> list[EditAction] | None:
    """Shortest edit path (<=2 steps) from ``state`` to ``target_structure``."""
    if state.structure == target_structure:
        return []
    for a in valid_structural_edits(state.structure):
        s1 = apply_edit(state, a)
        if s1.structure == target_structure:
            return [a]
        for b in valid_structural_edits(s1.structure):
            if apply_edit(s1, b).structure == target_structure:
                return [a, b]
    return None


def best_first_repair(
    env: ModelEditEnv,
    prm: ProcessRewardModel,
    *,
    max_depth: int = 3,
    beam_width: int = 3,
) -> SearchResult:
    """Search the edit space for the highest-PRM-potential structure and commit it.

    Args:
        env: The environment (provides ``observe_state`` previews + reward).
        prm: The process reward model scoring previewed observations.
        max_depth: Maximum edits to explore from the start.
        beam_width: Frontier width.

    Returns:
        A :class:`SearchResult` including the true terminal reward.
    """
    start = ModelState.initial(env.initial_structure)
    best_state = start
    best_pot = prm.potential(env.observe_state(start))

    frontier = [start]
    seen = {start.structure}
    for _ in range(max_depth):
        scored: list[tuple[float, ModelState]] = []
        for state in frontier:
            for action in valid_structural_edits(state.structure):
                nxt = apply_edit(state, action)
                if nxt.structure in seen:
                    continue
                seen.add(nxt.structure)
                pot = prm.potential(env.observe_state(nxt))
                scored.append((pot, nxt))
                if pot > best_pot:
                    best_pot, best_state = pot, nxt
        if not scored:
            break
        scored.sort(key=lambda x: x[0], reverse=True)
        frontier = [s for _, s in scored[:beam_width]]

    path = _path_to(start, best_state.structure) or []
    actions: tuple[Action, ...] = (*path, Commit())

    env.reset()
    reward = 0.0
    for act in actions:
        _obs, reward, _done, _info = env.step(act)
    return SearchResult(
        actions=actions,
        committed_structure=best_state.structure,
        best_potential=best_pot,
        terminal_reward=reward,
    )


def one_shot_commit(env: ModelEditEnv) -> float:
    """Terminal reward of committing the initial structure immediately."""
    env.reset()
    _obs, reward, _done, _info = env.step(Commit())
    return reward
