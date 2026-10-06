"""Simulation core. Coordinates are (row, column); boundaries do not wrap."""

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from random import Random


class Color(IntEnum):
    EMPTY = 0
    RED = 1
    BLUE = 2


SPECIES = (Color.RED, Color.BLUE)


@dataclass(frozen=True)
class Rules:
    random_birth_chance: float = 0.2
    interior_dissolve_chance: float = 0.02
    growth_probability: float = 0.12
    isolated_death: float = 0.35
    conversion_probability: float = 0.5
    conversion_neighbors: int = 3
    diagonal_weight: float = 2.0
    capture_enabled: bool = True

    def __post_init__(self):
        for name in ("random_birth_chance", "interior_dissolve_chance", "growth_probability", "isolated_death",
                     "conversion_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not isinstance(self.conversion_neighbors, int) or not 1 <= self.conversion_neighbors <= 8:
            raise ValueError("conversion_neighbors must be an integer between 1 and 8")
        if not 0 < self.diagonal_weight < float("inf"):
            raise ValueError("diagonal_weight must be positive and finite")


class Board:
    """Separate grids for points and owners; empty owned cells are territory.

    Grids are readable. Use set_cell for scenario editing to preserve invariants.
    """

    def __init__(self, width: int, height: int):
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("Board dimensions must be positive integers")
        self.width, self.height = width, height
        self.points = [[Color.EMPTY for _ in range(width)] for _ in range(height)]
        self.owners = [[Color.EMPTY for _ in range(width)] for _ in range(height)]
        # Naturally dissolved points may never become points again, even if
        # their former contour is subsequently destroyed.
        self.dissolved = set()

    def cells(self):
        for row in range(self.height):
            for col in range(self.width):
                yield row, col

    def neighbors(self, row: int, col: int, diagonals: bool = True):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr == dc == 0) or (not diagonals and dr != 0 and dc != 0):
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    yield nr, nc

    def set_cell(self, row: int, col: int, point=Color.EMPTY, owner=None):
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IndexError("Cell is outside the board")
        point = Color(point)
        owner = point if owner is None else Color(owner)
        if point != Color.EMPTY and owner != point:
            raise ValueError("An occupied cell must belong to its point's color")
        if point != Color.EMPTY and (row, col) in self.dissolved:
            raise ValueError("A naturally dissolved cell cannot become a point again")
        self.points[row][col], self.owners[row][col] = point, owner

    def copy(self):
        board = Board(self.width, self.height)
        board.points = [row[:] for row in self.points]
        board.owners = [row[:] for row in self.owners]
        board.dissolved = set(self.dissolved)
        return board

    def statistics(self):
        return {
            color.name.lower(): {
                "points": sum(self.points[r][c] == color for r, c in self.cells()),
                "territory": sum(self.owners[r][c] == color for r, c in self.cells()),
                "empty_territory": sum(self.owners[r][c] == color and
                                       self.points[r][c] == Color.EMPTY for r, c in self.cells()),
            } for color in SPECIES
        }

    def to_ascii(self):
        symbols = {Color.EMPTY: ".", Color.RED: "r", Color.BLUE: "b"}
        return "\n".join("".join(
            self.points[r][c].name[0] if self.points[r][c] else symbols[self.owners[r][c]]
            for c in range(self.width)) for r in range(self.height))


@dataclass(frozen=True)
class StepResult:
    generation: int
    births: int
    deaths: int
    conversions: int
    captured_cells: int
    captured_points: int
    highlighted_cells: frozenset[tuple[int, int]] = field(default_factory=frozenset, repr=False)
    highlighted_points: frozenset[tuple[int, int]] = field(default_factory=frozenset, repr=False)
    dissolved_points: int = 0


class Simulation:
    def __init__(self, board: Board, rules: Rules | None = None, seed: int | None = None):
        self.board = board.copy()
        self.rules = rules or Rules()
        self.random = Random(seed)
        self.generation = 0
        self._captured_point_positions = set()

    @classmethod
    def random_field(cls, width=30, height=20, density=0.08, rules=None, seed=None):
        if not 0 <= density <= 1:
            raise ValueError("density must be between 0 and 1")
        sim = cls(Board(width, height), rules, seed)
        for r, c in sim.board.cells():
            if sim.random.random() < density:
                sim.board.set_cell(r, c, sim.random.choice(SPECIES))
        return sim

    def enclosed_cells(self, color: Color):
        """Four-connected cells unreachable from edges; own points are walls.

        Cells on the field edge always have access to the outside. Territories
        without points are traversable. Diagonal point chains close a contour.
        """
        color = Color(color)
        if color not in SPECIES:
            raise ValueError("Capture color must be RED or BLUE")
        board = self.board
        reached = set()
        queue = deque()
        for r, c in board.cells():
            if (r in (0, board.height - 1) or c in (0, board.width - 1)) and board.points[r][c] != color:
                reached.add((r, c))
                queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for nr, nc in board.neighbors(r, c, diagonals=False):
                if (nr, nc) not in reached and board.points[nr][nc] != color:
                    reached.add((nr, nc))
                    queue.append((nr, nc))
        return {(r, c) for r, c in board.cells()
                if board.points[r][c] != color and (r, c) not in reached}

    def _claims(self):
        claims = {color: self.enclosed_cells(color) for color in SPECIES}
        conflict = claims[Color.RED] & claims[Color.BLUE]
        return {color: claims[color] - conflict for color in SPECIES}

    def protected_points(self):
        """Boundary points of enclosed components containing an empty cell.

        Filled rings without empty interiors do not grant protection. Branches
        outside the contour are not protected merely by touching its boundary.
        """
        if not self.rules.capture_enabled:
            return set()
        protected = set()
        for color, area in self._claims().items():
            remaining = set(area)
            while remaining:
                start = remaining.pop()
                component, queue = {start}, deque([start])
                while queue:
                    r, c = queue.popleft()
                    for cell in self.board.neighbors(r, c, diagonals=False):
                        if cell in remaining:
                            remaining.remove(cell)
                            component.add(cell)
                            queue.append(cell)
                if any(self.board.points[r][c] == Color.EMPTY for r, c in component):
                    for r, c in component:
                        for nr, nc in self.board.neighbors(r, c, diagonals=False):
                            if self.board.points[nr][nc] == color:
                                protected.add((nr, nc))
        return protected

    def can_remove_interior(self, row: int, col: int, color: Color):
        return (row, col) in self.removable_points(color)

    def removable_points(self, color: Color):
        """Interior removals cannot connect enclosed cells to the exterior.

        The new empty cell has four neighbors that are either own walls or
        already enclosed empty cells. This connectivity test needs only one
        flood fill per species for the entire board, not one per candidate.
        """
        if not self.rules.capture_enabled:
            return set()
        enemy = Color.BLUE if color == Color.RED else Color.RED
        own_area, enemy_area = self.enclosed_cells(color), self.enclosed_cells(enemy)
        board, safe = self.board, set()
        for r, c in board.cells():
            if board.points[r][c] != color or (r, c) in enemy_area:
                continue
            neighbors = list(board.neighbors(r, c))
            if len(neighbors) != 8 or any(board.owners[nr][nc] != color for nr, nc in neighbors):
                continue
            if all(board.points[nr][nc] == color or (nr, nc) in own_area
                   for nr, nc in board.neighbors(r, c, diagonals=False)):
                safe.add((r, c))
        return safe

    def capture(self):
        """Capture points simultaneously, then recompute territory on final contours."""
        raw_claims = {color: self.enclosed_cells(color) for color in SPECIES}
        disputed = raw_claims[Color.RED] & raw_claims[Color.BLUE]
        claims = {color: raw_claims[color] - disputed for color in SPECIES}
        previous = [row[:] for row in self.board.owners]
        self._captured_point_positions = set()
        points = 0
        for color in SPECIES:
            for r, c in sorted(claims[color]):
                if self.board.points[r][c] not in (Color.EMPTY, color):
                    points += 1
                    self._captured_point_positions.add((r, c))
                    self.board.points[r][c] = color
        # Converted points can break the other species' contour. Re-evaluate
        # empty territory on this final geometry, without cascading point capture.
        final_claims = self._claims() if points else claims
        self.board.owners = [row[:] for row in self.board.points]
        for color in SPECIES:
            for r, c in final_claims[color] - disputed:
                if self.board.points[r][c] == Color.EMPTY:
                    self.board.owners[r][c] = color
        cells = sum(self.board.owners[r][c] != Color.EMPTY and
                    self.board.owners[r][c] != previous[r][c] for r, c in self.board.cells())
        return cells, points

    def dissolve_interior(self):
        """Sequential removals preserve closed contours, including without growth."""
        positions = set()
        if self.rules.capture_enabled and self.rules.interior_dissolve_chance > 0:
            for r, c in self.board.cells():
                color = self.board.points[r][c]
                if (color != Color.EMPTY and self.random.random() < self.rules.interior_dissolve_chance
                        and self.can_remove_interior(r, c, color)):
                    self.board.set_cell(r, c, Color.EMPTY, color)
                    self.board.dissolved.add((r, c))
                    positions.add((r, c))
        return positions

    def step(self, autonomous=True):
        """One autonomous phase; agents can intervene after this method returns."""
        if not autonomous:
            dissolved = self.dissolve_interior()
            self.generation += 1
            return StepResult(self.generation, 0, 0, 0, 0, 0,
                              frozenset(dissolved), frozenset(), len(dissolved))
        old, new, rules = self.board, self.board.copy(), self.rules
        protected = self.protected_points()
        converted_positions = set()
        self._captured_point_positions = set()
        births = deaths = conversions = 0
        for r, c in old.cells():
            counts = {color: 0 for color in SPECIES}
            weights = {color: 0.0 for color in SPECIES}
            for nr, nc in old.neighbors(r, c):
                color = old.points[nr][nc]
                if color in SPECIES:
                    counts[color] += 1
                    weights[color] += rules.diagonal_weight if nr != r and nc != c else 1
            point = old.points[r][c]
            if point == Color.EMPTY:
                if old.owners[r][c] != Color.EMPTY or (r, c) in old.dissolved:
                    continue
                total = sum(weights.values())
                # Independent local growth opportunities aggregated to one event.
                probability = 1 - (1 - rules.growth_probability) ** total
                born = Color.EMPTY
                if total and self.random.random() < probability:
                    born = Color.RED if self.random.random() * total < weights[Color.RED] else Color.BLUE
                if born:
                    new.set_cell(r, c, born)
                    births += 1
            else:
                enemy = Color.BLUE if point == Color.RED else Color.RED
                # Conversion takes precedence over isolated death.
                if (counts[enemy] >= rules.conversion_neighbors
                        and self.random.random() < rules.conversion_probability):
                    new.set_cell(r, c, enemy)
                    conversions += 1
                    converted_positions.add((r, c))
                elif ((r, c) not in protected and counts[point] == 0
                      and self.random.random() < rules.isolated_death):
                    new.set_cell(r, c, Color.EMPTY)
                    deaths += 1
        self.board = new
        # At most one spontaneous birth per turn, independent of field size.
        # Use old occupancy as well to avoid replacing a point killed this turn.
        if self.random.random() < rules.random_birth_chance:
            free = [(r, c) for r, c in new.cells() if old.owners[r][c] == Color.EMPTY
                    and old.points[r][c] == Color.EMPTY and new.points[r][c] == Color.EMPTY
                    and (r, c) not in new.dissolved]
            if free:
                r, c = self.random.choice(free)
                new.set_cell(r, c, self.random.choice(SPECIES))
                births += 1
        captured_cells, captured_points = self.capture() if rules.capture_enabled else (0, 0)
        if not rules.capture_enabled:
            self.board.owners = [row[:] for row in self.board.points]
        dissolved_positions = self.dissolve_interior()
        highlighted_points = frozenset(converted_positions | self._captured_point_positions)
        highlighted_cells = frozenset(dissolved_positions | {(r, c) for r, c in old.cells()
                                     if self.board.points[r][c] == Color.EMPTY
                                     and self.board.owners[r][c] != Color.EMPTY
                                     and self.board.owners[r][c] != old.owners[r][c]})
        self.generation += 1
        return StepResult(self.generation, births, deaths, conversions, captured_cells,
                          captured_points, highlighted_cells, highlighted_points, len(dissolved_positions))
