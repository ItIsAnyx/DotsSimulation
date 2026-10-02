"""Baselines and a tiny trainable masked policy, using only the standard library."""

from collections import Counter
from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
from random import Random

from .game import Action, ActionKind, PASS
from .model import Color, SPECIES


LEGACY_FEATURE_NAMES = (
    "pass", "place", "remove", "capture",
    "near_own_points", "near_enemy_points", "near_own_empty", "near_enemy_empty", "near_free",
    "radius2_own_points", "radius2_enemy_points", "radius2_own_empty",
    "edge_distance", "enemy_pressure", "budget", "income",
    "own_territory", "enemy_territory", "score_difference", "round_progress",
)
PATCH_RADIUS = 4
FEATURE_NAMES = LEGACY_FEATURE_NAMES + tuple(
    f"patch_{dr}_{dc}_{channel}" for dr in range(-4, 5) for dc in range(-4, 5)
    for channel in ("point", "owner", "valid")) + tuple(
    f"overview_{r}_{c}_{channel}" for r in range(4) for c in range(4)
    for channel in ("own_points", "enemy_points", "own_empty", "enemy_empty")) + (
    "action_row", "action_col", "enemy_budget", "own_touched", "enemy_touched", "phase")


def action_features(game, actions, color=None, spatial=True):
    """Color-relative observations; neither color has a privileged encoding."""
    board = game.simulation.board
    color = game.active_color if color is None else color
    enemy = Color.BLUE if color == Color.RED else Color.RED
    stats = board.statistics()
    area = board.width * board.height
    player = game.players[color]
    global_features = [math.log1p(player.budget) / math.log1p(area * game.round_limit), player.income / area,
                       stats[color.name.lower()]["territory"] / area,
                       stats[enemy.name.lower()]["territory"] / area,
                       math.tanh((player.score - game.players[enemy].score) / 100),
                       game.round / game.round_limit]
    result = []
    overview = [[0.0] * 4 for _ in range(16)]
    sizes = [0] * 16
    if spatial:
        for r, c in board.cells():
            index = min(3, r * 4 // board.height) * 4 + min(3, c * 4 // board.width)
            sizes[index] += 1
            point, owner = board.points[r][c], board.owners[r][c]
            k = (0 if point == color else 1 if point == enemy else
                 2 if owner == color else 3 if owner == enemy else None)
            if k is not None:
                overview[index][k] += 1
    overview_flat = [v / max(1, sizes[i]) for i, values in enumerate(overview) for v in values]
    kinds = (ActionKind.PASS, ActionKind.PLACE, ActionKind.REMOVE, ActionKind.CAPTURE)
    for action in actions:
        features = [float(action.kind == kind) for kind in kinds]
        local, radius = [0.0] * 5, [0.0] * 3
        edge = pressure = 0.0
        if action.kind != ActionKind.PASS:
            r, c = action.row, action.col
            for nr, nc in board.neighbors(r, c):
                point, owner = board.points[nr][nc], board.owners[nr][nc]
                index = (0 if point == color else 1 if point == enemy else
                         2 if owner == color else 3 if owner == enemy else 4)
                local[index] += 1 / 8
            for nr in range(max(0, r - 2), min(board.height, r + 3)):
                for nc in range(max(0, c - 2), min(board.width, c + 3)):
                    if nr == r and nc == c:
                        continue
                    point, owner = board.points[nr][nc], board.owners[nr][nc]
                    if point == color:
                        radius[0] += 1 / 24
                    elif point == enemy:
                        radius[1] += 1 / 24
                    elif owner == color:
                        radius[2] += 1 / 24
            edge = min(r, c, board.height - 1 - r, board.width - 1 - c) / max(1, min(board.width, board.height) / 2)
            pressure = min(1.0, local[1] * 8 / game.simulation.rules.conversion_neighbors)
        x = features + local + radius + [edge, pressure] + global_features
        if spatial:
            center = (board.height // 2, board.width // 2) if action.kind == ActionKind.PASS else (action.row, action.col)
            for dr in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                for dc in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                    nr, nc = center[0] + dr, center[1] + dc
                    valid = 0 <= nr < board.height and 0 <= nc < board.width
                    if valid:
                        point, owner = board.points[nr][nc], board.owners[nr][nc]
                        x.extend([float(point == color) - float(point == enemy),
                                  float(owner == color) - float(owner == enemy), 1.0])
                    else:
                        x.extend([0.0, 0.0, 0.0])
            x.extend(overview_flat)
            x.extend([center[0] / max(1, board.height - 1), center[1] / max(1, board.width - 1),
                      math.log1p(game.players[enemy].budget) / math.log1p(area * game.round_limit),
                      len(game.touched[color]) / area, len(game.touched[enemy]) / area, game.phase_index / 2])
        result.append(x)
    return result


def state_features(game, color):
    x = action_features(game, [PASS], color=color)[0]
    x[:4] = [0.0] * 4
    return x


class RandomAgent:
    def __init__(self, seed=0):
        self.random = Random(seed)

    def choose(self, game, legal):
        # Choose action TYPE first, otherwise PASS becomes vanishingly unlikely
        # on large fields and random agents spend every resource immediately.
        groups = {}
        for action in legal:
            groups.setdefault(action.kind, []).append(action)
        return self.random.choice(groups[self.random.choice(list(groups))])


class HeuristicAgent:
    """Inspectable baseline: prioritize capture, threatened patches and income."""
    def choose(self, game, legal):
        features = action_features(game, legal, spatial=False)
        candidates = []
        for action, x in zip(legal, features):
            if action.kind == ActionKind.PASS:
                value = 0
            elif action.kind == ActionKind.CAPTURE:
                value = 5
            elif action.kind == ActionKind.PLACE:
                value = 6 * x[5] - 0.6  # Patch only where enemy pressure exists.
            else:
                value = 1 - 8 * x[11]  # Create income away from enemy points.
            candidates.append(value)
        return legal[max(range(len(legal)), key=candidates.__getitem__)]


@dataclass
class Decision:
    features: list
    offsets: list
    selected: int
    reward: float = 0.0
    round: int = 0
    cached_gradient: tuple | None = None
    choice_count: int = 0
    observation: array | None = None
    baseline: float | None = None


class ValueNetwork:
    """Separate learned state-value model; no hand-written tactical targets."""
    def __init__(self, seed=0, hidden=16):
        rng = Random(seed)
        scale = 1 / math.sqrt(len(FEATURE_NAMES))
        self.hidden = hidden
        self.w1 = [[rng.uniform(-scale, scale) for _ in FEATURE_NAMES] for _ in range(hidden)]
        self.b1, self.w2, self.bias = [0.0] * hidden, [0.0] * hidden, 0.0

    def forward(self, x):
        h = [math.tanh(sum(w * v for w, v in zip(row, x)) + bias)
             for row, bias in zip(self.w1, self.b1)]
        return sum(w * v for w, v in zip(self.w2, h)) + self.bias, h

    def predict(self, x):
        return self.forward(x)[0]

    def fit(self, samples, learning_rate):
        if not samples:
            return
        gw1 = [[0.0] * len(FEATURE_NAMES) for _ in range(self.hidden)]
        gb1, gw2, gb = [0.0] * self.hidden, [0.0] * self.hidden, 0.0
        for x, target in samples:
            prediction, h = self.forward(x)
            error = prediction - target
            gb += error
            for j in range(self.hidden):
                gw2[j] += error * h[j]
                delta = error * self.w2[j] * (1 - h[j] ** 2)
                gb1[j] += delta
                for k, v in enumerate(x):
                    gw1[j][k] += delta * v
        norm = math.sqrt(gb * gb + sum(v * v for row in gw1 for v in row)
                         + sum(v * v for v in gb1 + gw2)) / len(samples)
        rate = learning_rate / len(samples) / max(1.0, norm)
        self.bias -= rate * gb
        for j in range(self.hidden):
            self.b1[j] -= rate * gb1[j]
            self.w2[j] -= rate * gw2[j]
            for k in range(len(FEATURE_NAMES)):
                self.w1[j][k] -= rate * gw1[j][k]

    def as_dict(self):
        return dict(hidden=self.hidden, w1=self.w1, b1=self.b1, w2=self.w2, bias=self.bias)


class TinyPolicy:
    """Spatial scorer plus a separate Monte-Carlo-trained value network.

    Softmax is over proposed legal candidates only. Offsets
    compensate the number of coordinates per action type, so PASS remains
    a meaningful option even when thousands of placements are possible.
    """
    def __init__(self, seed=0, hidden=16):
        self.random = Random(seed)
        self.hidden = hidden
        scale = 1 / math.sqrt(len(FEATURE_NAMES))
        self.w1 = [[self.random.uniform(-scale, scale) for _ in FEATURE_NAMES] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [self.random.uniform(-0.2, 0.2) for _ in range(hidden)]
        self.episodes = 0
        self.training_version = 3
        self.value = ValueNetwork(seed + 10000, hidden)

    def forward(self, features, offsets=None):
        activations, logits = [], []
        for x in features:
            h = [math.tanh(sum(w * value for w, value in zip(row, x)) + bias)
                 for row, bias in zip(self.w1, self.b1)]
            activations.append(h)
            logits.append(sum(w * value for w, value in zip(self.w2, h)))
        if offsets is not None:
            logits = [logit + offset for logit, offset in zip(logits, offsets)]
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        total = sum(weights)
        return [weight / total for weight in weights], activations

    def distribution(self, game, legal, active=False, candidate_limit=64):
        """Sample coordinates without geometric priors, then score with the policy.

        PASS is legal by default. Only explicit legacy active=True applies the
        old capture guard and surplus-budget prior. Type normalization prevents
        a larger number of coordinates from drowning out other action types.
        """
        candidates = list(legal)
        if active and any(a.kind == ActionKind.CAPTURE for a in candidates):
            candidates = [a for a in candidates if a.kind != ActionKind.PASS]
        if candidate_limit is not None:
            # The proposal sampler is independent of network weights. It keeps
            # every action type and samples coordinates anew at every decision.
            groups = {}
            for action in candidates:
                groups.setdefault(action.kind, []).append(action)
            candidates = []
            for group in groups.values():
                candidates.extend(group if len(group) <= candidate_limit else self.random.sample(group, candidate_limit))
        features = action_features(game, candidates)
        counts = Counter(action.kind for action in candidates)
        offsets = [-math.log(counts[action.kind]) for action in candidates]
        if active and len(candidates) > 1:
            player = game.players[game.active_color]
            surplus = player.budget / (player.income + 1)
            for i, action in enumerate(candidates):
                if action.kind == ActionKind.PASS:
                    offsets[i] -= math.log1p(math.sqrt(surplus))
        probabilities, _ = self.forward(features, offsets)
        return candidates, features, offsets, probabilities

    def decide(self, game, legal, greedy=False, active=False, candidate_limit=64):
        candidates, features, offsets, probabilities = self.distribution(game, legal, active, candidate_limit)
        if greedy:
            selected = max(range(len(candidates)), key=probabilities.__getitem__)
        else:
            selected, threshold, cumulative = len(candidates) - 1, self.random.random(), 0.0
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if threshold < cumulative:
                    selected = index
                    break
        observation = array('d', state_features(game, game.active_color))
        return candidates[selected], Decision(features, offsets, selected, round=game.round,
                                             observation=observation, baseline=self.value.predict(observation))

    def cache_gradient(self, decision):
        """Weights stay frozen throughout an episode, so cache their exact
        score gradient instead of retaining every candidate observation.
        This bounds trajectory memory when active turns contain many actions.
        """
        decision.choice_count = len(decision.features)
        if decision.choice_count > 1:
            w1, b1, w2 = self.gradient(decision)
            decision.cached_gradient = ([array('d', row) for row in w1], array('d', b1), array('d', w2))
        decision.features, decision.offsets = [], []

    def gradient(self, decision):
        """Exact gradient of log pi(selected | state), independently testable."""
        if decision.cached_gradient is not None:
            return decision.cached_gradient
        probabilities, activations = self.forward(decision.features, decision.offsets)
        gw1 = [[0.0] * len(FEATURE_NAMES) for _ in range(self.hidden)]
        gb1, gw2 = [0.0] * self.hidden, [0.0] * self.hidden
        for index, (x, h, probability) in enumerate(zip(decision.features, activations, probabilities)):
            coefficient = (1.0 if index == decision.selected else 0.0) - probability
            for j in range(self.hidden):
                gw2[j] += coefficient * h[j]
                delta = coefficient * self.w2[j] * (1 - h[j] ** 2)
                gb1[j] += delta
                for k, value in enumerate(x):
                    gw1[j][k] += delta * value
        return gw1, gb1, gw2

    def update(self, trajectory, learning_rate=0.01, gamma=0.995, reward_scale=20):
        """REINFORCE with a lagged baseline and clipped batch gradient.

        The policy stores a baseline from previous episodes; observations
        from the current batch do not define their own baseline.
        """
        baseline = getattr(self, "baseline", 0.0)
        returns, value = [], 0.0
        next_round = trajectory[-1].round if trajectory else 0
        for decision in reversed(trajectory):
            # Hundreds of actions in one turn do not advance game time.
            discount = gamma ** max(0, next_round - decision.round)
            value = decision.reward / reward_scale + discount * value
            returns.append(value)
            next_round = decision.round
        returns.reverse()
        if not trajectory:
            self.episodes += 1
            return 0.0
        gw1 = [[0.0] * len(FEATURE_NAMES) for _ in range(self.hidden)]
        gb1, gw2 = [0.0] * self.hidden, [0.0] * self.hidden
        for decision, value in zip(trajectory, returns):
            if (decision.choice_count or len(decision.features)) <= 1:
                continue  # A forced PASS has zero policy gradient.
            gradients = self.gradient(decision)
            advantage = value - (decision.baseline if decision.baseline is not None else baseline)
            for j in range(self.hidden):
                gb1[j] += gradients[1][j] * advantage
                gw2[j] += gradients[2][j] * advantage
                for k in range(len(FEATURE_NAMES)):
                    gw1[j][k] += gradients[0][j][k] * advantage
        norm = math.sqrt(sum(v * v for row in gw1 for v in row) +
                         sum(v * v for v in gb1) + sum(v * v for v in gw2)) / len(trajectory)
        multiplier = learning_rate / len(trajectory) / max(1.0, norm)
        for j in range(self.hidden):
            self.b1[j] += multiplier * gb1[j]
            self.w2[j] += multiplier * gw2[j]
            for k in range(len(FEATURE_NAMES)):
                self.w1[j][k] += multiplier * gw1[j][k]
        self.baseline = 0.95 * baseline + 0.05 * sum(returns) / len(returns)
        self.value.fit([(d.observation, target) for d, target in zip(trajectory, returns)
                        if d.observation is not None], learning_rate)
        self.episodes += 1
        self.training_version = 3
        return norm

    def as_dict(self):
        return dict(features=list(FEATURE_NAMES), hidden=self.hidden, w1=self.w1,
                    b1=self.b1, w2=self.w2, episodes=self.episodes,
                    baseline=getattr(self, "baseline", 0.0), training_version=self.training_version,
                    value=self.value.as_dict())

    @classmethod
    def from_dict(cls, data, seed=0):
        legacy = data["features"] == list(LEGACY_FEATURE_NAMES)
        if not legacy and data["features"] != list(FEATURE_NAMES):
            raise ValueError("Checkpoint features do not match this policy")
        hidden = data["hidden"]
        if not isinstance(hidden, int) or not 1 <= hidden <= 256:
            raise ValueError("Invalid hidden layer size")
        policy = cls(seed, hidden)
        input_size = len(LEGACY_FEATURE_NAMES) if legacy else len(FEATURE_NAMES)
        if (len(data["w1"]) != hidden or any(len(row) != input_size for row in data["w1"])
                or len(data["b1"]) != hidden or len(data["w2"]) != hidden):
            raise ValueError("Invalid checkpoint weight shapes")
        if not all(math.isfinite(v) for row in data["w1"] for v in row) or not all(
                math.isfinite(v) for v in data["b1"] + data["w2"]):
            raise ValueError("Checkpoint contains non-finite weights")
        policy.w1 = [row[:] + [0.0] * (len(FEATURE_NAMES) - input_size) for row in data["w1"]]
        policy.b1, policy.w2 = data["b1"][:], data["w2"][:]
        policy.episodes = data.get("episodes", 0)
        policy.training_version = data.get("training_version", 1)
        # A baseline trained on per-decision discounts and raw score is not
        # compatible with zero-sum, round-discounted returns.
        policy.baseline = data.get("baseline", 0.0) if policy.training_version == 3 else 0.0
        if "value" in data and not legacy:
            value = data["value"]
            if (value["hidden"] != hidden or len(value["w1"]) != hidden or
                    any(len(row) != len(FEATURE_NAMES) for row in value["w1"]) or
                    len(value["b1"]) != hidden or len(value["w2"]) != hidden):
                raise ValueError("Invalid value-network shape")
            values = [v for row in value["w1"] for v in row] + value["b1"] + value["w2"] + [value["bias"]]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite value-network weights")
            policy.value.w1 = [row[:] for row in value["w1"]]
            policy.value.b1, policy.value.w2, policy.value.bias = value["b1"][:], value["w2"][:], value["bias"]
        return policy


class NeuralAgent:
    def __init__(self, policy=None, seed=0, active=False):
        self.policy = policy or TinyPolicy(seed)
        self.active = active

    def choose(self, game, legal):
        return self.policy.decide(game, legal, active=self.active)[0]


def save_policies(path, policies, metadata=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(version=1, metadata=metadata or {},
        policies={c.name.lower(): policies[c].as_dict() for c in SPECIES}), indent=2), encoding="utf-8")


def load_policies(path, seed=0):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
    return {color: TinyPolicy.from_dict(data["policies"][color.name.lower()], seed + int(color))
            for color in SPECIES}
