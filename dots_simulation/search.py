"""Inference-only beam search over legal continuations of the current turn."""

from dataclasses import dataclass
import math
from random import Random

from .agents import TinyPolicy, state_features
from .game import ActionKind, PASS
from .model import Color


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 4
    beam: int = 4
    branches: int = 5
    max_nodes: int = 100

    def __post_init__(self):
        if not (1 <= self.depth <= 8 and 1 <= self.beam <= 16 and
                2 <= self.branches <= 16 and 1 <= self.max_nodes <= 2000):
            raise ValueError("Search: depth 1..8, beam 1..16, branches 2..16, nodes 1..2000")


class SearchAgent:
    """Replan after each real action; never modify weights or the live game.

    Proposals use policy probabilities plus sampled alternatives. Ranking uses
    actual zero-sum game reward / 20 plus the learned Monte Carlo value.
    There are no bonuses for geometry, income, captures, or action spending.
    Opponent and automaton phases are outside this bounded search horizon.
    """

    def __init__(self, policy=None, seed=0, config=None):
        self.policy = policy or TinyPolicy(seed)
        self.random = Random(seed)
        self.config = config or SearchConfig()
        self.last_sequence = ()
        self.last_nodes = 0
        self.last_value = 0.0

    def _proposals(self, game, legal):
        # Shortlist RNG is local to search, leaving inference/training RNG intact.
        old_random = self.policy.random
        self.policy.random = self.random
        try:
            actions, _, _, probabilities = self.policy.distribution(game, legal, active=False)
        finally:
            self.policy.random = old_random
        ranked = sorted(zip(actions, probabilities), key=lambda pair: pair[1], reverse=True)
        nonpass = [(a, p) for a, p in ranked if a != PASS]
        selected = nonpass[:max(1, (self.config.branches - 1) // 2)]
        remaining = [pair for pair in nonpass if pair not in selected]
        while remaining and len(selected) < self.config.branches - 1:
            index = self.random.choices(range(len(remaining)), [p for _, p in remaining])[0]
            selected.append(remaining.pop(index))
        selected.append((PASS, probabilities[actions.index(PASS)]))
        return selected

    def choose(self, game, legal):
        if legal == [PASS]:
            self.last_sequence, self.last_nodes, self.last_value = (PASS,), 0, 0.0
            return PASS
        color = game.active_color
        enemy = Color.RED if color == Color.BLUE else Color.BLUE
        # state, sequence, accumulated reward, log likelihood
        frontier = [(game.clone(), (), 0.0, 0.0)]
        best = None
        nodes = 0
        for _ in range(self.config.depth):
            continuations = []
            for state, sequence, reward, likelihood in frontier:
                for action, probability in self._proposals(state, state.legal_actions()):
                    if nodes >= self.config.max_nodes:
                        break
                    child = state.clone()
                    rewards = child.apply_action(action)
                    ended = action.kind == ActionKind.PASS
                    if ended:
                        terminal = child.finish_agent_turn()
                        rewards = {c: rewards[c] + terminal[c] for c in rewards}
                    total = reward + (rewards[color] - rewards[enemy]) / 20
                    path = sequence + (action,)
                    logp = likelihood + math.log(max(probability, 1e-300))
                    value = total + (0.0 if child.done else self.policy.value.predict(state_features(child, color)))
                    rank = (value, logp / len(path))
                    nodes += 1
                    if best is None or rank > best[0]:
                        best = (rank, path)
                    if not ended:
                        continuations.append((rank, child, path, total, logp))
                if nodes >= self.config.max_nodes:
                    break
            continuations.sort(key=lambda item: item[0], reverse=True)
            frontier = [item[1:] for item in continuations[:self.config.beam]]
            if not frontier or nodes >= self.config.max_nodes:
                break
        self.last_nodes = nodes
        self.last_sequence = best[1] if best else (PASS,)
        self.last_value = best[0][0] if best else 0.0
        return self.last_sequence[0]
