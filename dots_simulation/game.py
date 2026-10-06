"""Phase controller, legal actions and score accounting, independent of agents."""

from dataclasses import dataclass, field, replace
from enum import Enum

from .model import Board, Color, Simulation, SPECIES, StepResult


class ActionKind(str, Enum):
    PASS = "pass"
    PLACE = "place"
    REMOVE = "remove"
    CAPTURE = "capture"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    row: int = -1
    col: int = -1

    @property
    def cost(self):
        return 0 if self.kind == ActionKind.PASS else 3 if self.kind == ActionKind.CAPTURE else 1


PASS = Action(ActionKind.PASS)


@dataclass
class Player:
    score: int = 0
    budget: int = 0
    income: int = 0
    spent: int = 0


@dataclass
class PhaseResult:
    phase: str
    rewards: dict
    action_count: int = 0
    last_action: Action | None = None
    automaton: StepResult | None = None
    highlighted_cells: frozenset = field(default_factory=frozenset, repr=False)
    highlighted_points: frozenset = field(default_factory=frozenset, repr=False)


class Game:
    """A round is automaton -> one player turn -> other player turn.

    A turn contains paid actions until PASS or exhaustion. Cells cannot be
    touched twice by the same player in a round. Budget carries over; income
    is sampled once immediately after the automaton phase.
    """

    def __init__(self, simulation: Simulation, agents_enabled=True, round_limit=200, victory_reward=50,
                 automaton_rounds=None):
        if round_limit < 1:
            raise ValueError("round_limit must be positive")
        if automaton_rounds is not None and (not isinstance(automaton_rounds, int) or automaton_rounds < 0):
            raise ValueError("automaton_rounds must be nonnegative or None")
        self.automaton_rounds = automaton_rounds
        self.simulation = simulation
        self.agents_enabled = agents_enabled
        self.round_limit = round_limit
        self.victory_reward = victory_reward
        self.players = {color: Player() for color in SPECIES}
        self.round = 0
        self.phase_index = 0
        self.order = (Color.BLUE, Color.RED)
        self.done = False
        self.winner = Color.EMPTY
        self.touched = {color: set() for color in SPECIES}
        # First neutral-cell reward is global per coordinate per episode.
        self.rewarded_empty = {(r, c) for r, c in simulation.board.cells()
                               if simulation.board.owners[r][c] != Color.EMPTY}
        self.last_result = None

    def clone(self):
        """Independent search state, including budgets, history and RNG state."""
        simulation = Simulation(self.simulation.board, self.simulation.rules)
        simulation.generation = self.simulation.generation
        simulation.random.setstate(self.simulation.random.getstate())
        clone = Game(simulation, self.agents_enabled, self.round_limit, self.victory_reward, self.automaton_rounds)
        clone.players = {color: replace(player) for color, player in self.players.items()}
        clone.round, clone.phase_index, clone.order = self.round, self.phase_index, self.order
        clone.done, clone.winner = self.done, self.winner
        clone.touched = {color: set(cells) for color, cells in self.touched.items()}
        clone.rewarded_empty = set(self.rewarded_empty)
        return clone

    @property
    def automaton_active(self):
        """Whether the upcoming maintenance phase includes autonomous changes."""
        return self.automaton_rounds is None or self.round < self.automaton_rounds

    @property
    def active_color(self):
        return None if self.phase_index == 0 else self.order[self.phase_index - 1]

    @property
    def next_phase(self):
        if self.done:
            return "Партия завершена"
        if self.phase_index == 0:
            return "Автомат" if self.automaton_active else "Растворение и доход"
        return "Синий агент" if self.active_color == Color.BLUE else "Красный агент"

    def _score_changes(self, before: Board):
        after = self.simulation.board
        rewards = {color: 0 for color in SPECIES}
        for r, c in before.cells():
            old, new = before.owners[r][c], after.owners[r][c]
            if old == new:
                continue  # Own PLACE / REMOVE / natural dissolution is not capture.
            if old != Color.EMPTY:
                rewards[old] -= 2 if before.points[r][c] != Color.EMPTY else 5
            if new != Color.EMPTY:
                if old != Color.EMPTY:
                    rewards[new] += 5
                elif (r, c) not in self.rewarded_empty:
                    rewards[new] += 5
                    self.rewarded_empty.add((r, c))
        for color in SPECIES:
            self.players[color].score += rewards[color]
        return rewards

    def _finish(self):
        if self.done or self.round < self.round_limit:
            return {color: 0 for color in SPECIES}
        self.done = True
        stats = self.simulation.board.statistics()
        red, blue = stats["red"]["territory"], stats["blue"]["territory"]
        rewards = {color: 0 for color in SPECIES}
        if red != blue:
            self.winner = Color.RED if red > blue else Color.BLUE
            loser = Color.BLUE if self.winner == Color.RED else Color.RED
            # Snapshot both pre-terminal scores. The winner finishes positive
            # and the loser negative even when intermediate scores are large.
            rewards[self.winner] = 2 * abs(self.players[self.winner].score) + self.victory_reward
            rewards[loser] = -(2 * abs(self.players[loser].score) + self.victory_reward)
            for color in SPECIES:
                self.players[color].score += rewards[color]
        return rewards

    def advance_automaton(self):
        if self.done or self.phase_index != 0:
            raise ValueError("It is not the automaton phase")
        before = self.simulation.board.copy()
        autonomous = self.automaton_active
        events = self.simulation.step(autonomous=autonomous)
        self.round += 1
        self.order = (Color.BLUE, Color.RED) if self.round % 2 else (Color.RED, Color.BLUE)
        rewards = self._score_changes(before)
        stats = self.simulation.board.statistics()
        for color in SPECIES:
            p = self.players[color]
            p.income = stats[color.name.lower()]["empty_territory"]
            p.budget += p.income
            p.spent = 0
            self.touched[color].clear()
        self.phase_index = 1 if self.agents_enabled else 0
        if not self.agents_enabled:
            terminal = self._finish()
            rewards = {c: rewards[c] + terminal[c] for c in SPECIES}
        self.last_result = PhaseResult("Автомат" if autonomous else "Растворение и доход", rewards, automaton=events,
                                       highlighted_cells=events.highlighted_cells,
                                       highlighted_points=events.highlighted_points)
        return self.last_result

    def legal_actions(self, color=None):
        color = self.active_color if color is None else color
        if self.done or color not in SPECIES or color != self.active_color:
            return [PASS]
        board, budget = self.simulation.board, self.players[color].budget
        if budget < 1:
            return [PASS]
        enemy = Color.BLUE if color == Color.RED else Color.RED
        removable = self.simulation.removable_points(color)
        actions = [PASS]
        for r, c in board.cells():
            if (r, c) in self.touched[color]:
                continue
            point, owner = board.points[r][c], board.owners[r][c]
            if owner == Color.EMPTY and point == Color.EMPTY and (r, c) not in board.dissolved:
                actions.append(Action(ActionKind.PLACE, r, c))
            elif owner == color:
                if point == Color.EMPTY and (r, c) not in board.dissolved:
                    actions.append(Action(ActionKind.PLACE, r, c))
                elif point == color:
                    if (r, c) in removable:
                        actions.append(Action(ActionKind.REMOVE, r, c))
            elif owner == enemy and point != Color.EMPTY and budget >= 3:
                attackers = sum(board.points[nr][nc] == color for nr, nc in board.neighbors(r, c))
                if attackers >= self.simulation.rules.conversion_neighbors:
                    actions.append(Action(ActionKind.CAPTURE, r, c))
        return actions

    def is_legal_action(self, action: Action):
        """Validate only the chosen action, without rebuilding the entire mask."""
        color, board = self.active_color, self.simulation.board
        if self.done or color is None:
            return False
        if action.kind == ActionKind.PASS:
            return action == PASS
        r, c = action.row, action.col
        if not (0 <= r < board.height and 0 <= c < board.width):
            return False
        if self.players[color].budget < action.cost or (r, c) in self.touched[color]:
            return False
        point, owner = board.points[r][c], board.owners[r][c]
        if action.kind == ActionKind.PLACE:
            return owner in (Color.EMPTY, color) and point == Color.EMPTY and (r, c) not in board.dissolved
        if action.kind == ActionKind.REMOVE:
            return owner == color and point == color and self.simulation.can_remove_interior(r, c, color)
        if action.kind == ActionKind.CAPTURE:
            enemy = Color.BLUE if color == Color.RED else Color.RED
            return (owner == enemy and point == enemy and
                    sum(board.points[nr][nc] == color for nr, nc in board.neighbors(r, c))
                    >= self.simulation.rules.conversion_neighbors)
        return False

    def apply_action(self, action: Action):
        if not self.is_legal_action(action):
            raise ValueError("Illegal action in the current phase")
        color = self.active_color
        before = self.simulation.board.copy()
        if action.kind == ActionKind.PASS:
            return {c: 0 for c in SPECIES}
        r, c = action.row, action.col
        if action.kind in (ActionKind.PLACE, ActionKind.CAPTURE):
            if before.points[r][c] == Color.EMPTY and action.kind == ActionKind.CAPTURE:
                self.simulation.board.set_cell(r, c, Color.EMPTY, color)
            else:
                self.simulation.board.set_cell(r, c, color)
        else:
            self.simulation.board.set_cell(r, c, Color.EMPTY, color)
        self.players[color].budget -= action.cost
        self.players[color].spent += action.cost
        self.touched[color].add((r, c))
        if self.simulation.rules.capture_enabled:
            self.simulation.capture()
        return self._score_changes(before)

    def finish_agent_turn(self):
        if self.done or self.phase_index == 0:
            raise ValueError("It is not an agent phase")
        self.phase_index = 2 if self.phase_index == 1 else 0
        return self._finish() if self.phase_index == 0 else {c: 0 for c in SPECIES}

    def advance(self, agents=None):
        """Synchronous convenience API for tests, CLI, and evaluation."""
        if self.done:
            raise ValueError("The episode is finished")
        if self.phase_index == 0:
            return self.advance_automaton()
        color, before = self.active_color, self.simulation.board.copy()
        phase = self.next_phase
        rewards = {c: 0 for c in SPECIES}
        count, last = 0, PASS
        while True:
            legal = self.legal_actions()
            last = agents[color].choose(self, legal) if agents else PASS
            reward = self.apply_action(last)
            for c in SPECIES:
                rewards[c] += reward[c]
            if last.kind == ActionKind.PASS:
                break
            count += 1
        terminal = self.finish_agent_turn()
        for c in SPECIES:
            rewards[c] += terminal[c]
        after = self.simulation.board
        self.last_result = PhaseResult(phase, rewards, count, last,
            highlighted_cells=frozenset((r, c) for r, c in after.cells()
                if after.points[r][c] == Color.EMPTY and after.owners[r][c] != Color.EMPTY
                and (before.owners[r][c] != after.owners[r][c] or before.points[r][c] != Color.EMPTY)),
            highlighted_points=frozenset((r, c) for r, c in after.cells()
                if after.points[r][c] != Color.EMPTY and before.points[r][c] != after.points[r][c]))
        return self.last_result
