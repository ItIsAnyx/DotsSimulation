import unittest

from dots_simulation import Board, Color, Rules, Simulation


def quiet_rules(**overrides):
    values = dict(random_birth_chance=0, interior_dissolve_chance=0, growth_probability=0, isolated_death=0,
                  conversion_probability=0, capture_enabled=False)
    values.update(overrides)
    return Rules(**values)


class ModelTests(unittest.TestCase):
    def test_points_and_empty_territory_are_separate(self):
        board = Board(3, 3)
        board.set_cell(1, 1, owner=Color.RED)
        self.assertEqual(board.statistics()["red"],
                         dict(points=0, territory=1, empty_territory=1))
        with self.assertRaises(ValueError):
            board.set_cell(0, 0, Color.BLUE, Color.RED)

    def test_death_releases_unenclosed_cell(self):
        board = Board(3, 3)
        board.set_cell(1, 1, Color.RED)
        sim = Simulation(board, quiet_rules(isolated_death=1))
        self.assertEqual(sim.step().deaths, 1)
        self.assertEqual(sim.board.points[1][1], Color.EMPTY)
        self.assertEqual(sim.board.owners[1][1], Color.EMPTY)

    def test_growth_uses_old_field_without_chain_reaction(self):
        board = Board(5, 5)
        board.set_cell(2, 2, Color.RED)
        sim = Simulation(board, quiet_rules(growth_probability=1), seed=1)
        self.assertEqual(sim.step().births, 8)
        self.assertEqual(sim.board.points[0][0], Color.EMPTY)
        self.assertEqual(sim.board.statistics()["red"]["points"], 9)

    def test_conversion_requires_three_attackers_regardless_of_defenders(self):
        board = Board(5, 5)
        board.set_cell(2, 2, Color.RED)
        for cell in ((1, 1), (1, 2), (1, 3)):
            board.set_cell(*cell, Color.BLUE)
        sim = Simulation(board, quiet_rules(conversion_probability=1))
        sim.step()
        self.assertEqual(sim.board.points[2][2], Color.BLUE)
        self.assertEqual(sim.board.owners[2][2], Color.BLUE)
        board.set_cell(3, 2, Color.RED)
        sim = Simulation(board, quiet_rules(conversion_probability=1))
        sim.step()
        self.assertEqual(sim.board.points[2][2], Color.BLUE)
        board.set_cell(1, 1)
        sim = Simulation(board, quiet_rules(conversion_probability=1))
        sim.step()
        self.assertEqual(sim.board.points[2][2], Color.RED)

    def test_diagonal_ring_captures_enemy(self):
        board = Board(5, 5)
        for cell in ((1, 2), (2, 1), (2, 3), (3, 2)):
            board.set_cell(*cell, Color.RED)
        board.set_cell(2, 2, Color.BLUE)
        sim = Simulation(board)
        self.assertEqual(sim.enclosed_cells(Color.RED), {(2, 2)})
        self.assertEqual(sim.capture(), (1, 1))
        self.assertEqual(sim.board.points[2][2], Color.RED)
        self.assertEqual(sim.capture(), (0, 0))

    def test_open_ring_does_not_capture(self):
        board = Board(5, 5)
        for cell in ((1, 2), (2, 1), (2, 3)):
            board.set_cell(*cell, Color.RED)
        self.assertEqual(Simulation(board).enclosed_cells(Color.RED), set())

    def test_territory_without_points_does_not_form_walls(self):
        board = Board(5, 5)
        for cell in ((1, 2), (2, 1), (2, 3), (3, 2)):
            board.set_cell(*cell, owner=Color.RED)
        self.assertEqual(Simulation(board).capture(), (0, 0))

    def test_conflicting_nested_claims_leave_center_unchanged(self):
        board = Board(7, 7)
        for r, c in board.cells():
            if r in (1, 5) and 1 <= c <= 5 or c in (1, 5) and 1 <= r <= 5:
                board.set_cell(r, c, Color.RED)
        for cell in ((2, 3), (3, 2), (3, 4), (4, 3)):
            board.set_cell(*cell, Color.BLUE)
        sim = Simulation(board)
        sim.capture()
        self.assertEqual(sim.board.owners[3][3], Color.EMPTY)

    def test_no_wrap_and_small_boards(self):
        board = Board(3, 3)
        self.assertEqual(len(list(board.neighbors(0, 0))), 3)
        for width, height in ((1, 1), (1, 5), (5, 1)):
            sim = Simulation.random_field(width, height, density=0, seed=1)
            self.assertEqual(sim.capture(), (0, 0))
            sim.step()

    def test_seed_reproduces_full_evolution_and_invariants(self):
        first = Simulation.random_field(15, 12, seed=123)
        second = Simulation.random_field(15, 12, seed=123)
        for _ in range(30):
            self.assertEqual(first.step(), second.step())
            self.assertEqual(first.board.to_ascii(), second.board.to_ascii())
            for r, c in first.board.cells():
                if first.board.points[r][c]:
                    self.assertEqual(first.board.points[r][c], first.board.owners[r][c])

    def test_invalid_parameters(self):
        for values in (dict(growth_probability=2), dict(diagonal_weight=0),
                       dict(conversion_neighbors=0), dict(random_birth_chance=1.1)):
            with self.assertRaises(ValueError):
                Rules(**values)
        with self.assertRaises(ValueError):
            Board(0, 2)

    def test_random_birth_is_one_per_turn_and_can_be_disabled(self):
        sim = Simulation(Board(20, 20), quiet_rules(random_birth_chance=1), seed=9)
        for count in range(1, 6):
            self.assertEqual(sim.step().births, 1)
            self.assertEqual(sum(s["points"] for s in sim.board.statistics().values()), count)
        sim.rules = quiet_rules(random_birth_chance=0)
        self.assertEqual(sim.step().births, 0)

    def test_random_birth_skips_full_field(self):
        board = Board(1, 1)
        board.set_cell(0, 0, Color.BLUE)
        sim = Simulation(board, quiet_rules(random_birth_chance=1))
        self.assertEqual(sim.step().births, 0)
        self.assertEqual(sim.board.points[0][0], Color.BLUE)

    def test_empty_area_protects_boundary_but_not_solid_cluster(self):
        board = Board(5, 5)
        ring = {(1, 2), (2, 1), (2, 3), (3, 2)}
        for cell in ring:
            board.set_cell(*cell, Color.RED)
        sim = Simulation(board, quiet_rules(capture_enabled=True, isolated_death=1))
        self.assertEqual(sim.protected_points(), ring)
        sim.step()
        self.assertEqual(sim.board.owners[2][2], Color.RED)
        self.assertTrue(all(sim.board.points[r][c] == Color.RED for r, c in ring))
        sim.board.set_cell(2, 2, Color.RED)
        self.assertEqual(sim.protected_points(), set())

    def test_broken_ring_releases_territory(self):
        board = Board(5, 5)
        for cell in ((1, 2), (2, 1), (2, 3), (3, 2)):
            board.set_cell(*cell, Color.RED)
        sim = Simulation(board, quiet_rules(capture_enabled=True))
        sim.capture()
        self.assertEqual(sim.board.owners[2][2], Color.RED)
        sim.board.set_cell(1, 2)
        sim.step()
        self.assertEqual(sim.board.owners[2][2], Color.EMPTY)

    def test_capture_highlights_expire_on_next_turn(self):
        board = Board(5, 5)
        for cell in ((1, 2), (2, 1), (2, 3), (3, 2)):
            board.set_cell(*cell, Color.RED)
        sim = Simulation(board, quiet_rules(capture_enabled=True))
        first = sim.step()
        self.assertEqual(first.highlighted_cells, frozenset({(2, 2)}))
        self.assertEqual(sim.step().highlighted_cells, frozenset())

    def test_local_conversion_highlights_enemy_point(self):
        board = Board(5, 5)
        board.set_cell(2, 2, Color.RED)
        for cell in ((1, 1), (1, 2), (1, 3)):
            board.set_cell(*cell, Color.BLUE)
        sim = Simulation(board, quiet_rules(conversion_probability=1))
        self.assertIn((2, 2), sim.step().highlighted_points)

    def test_protected_contour_can_be_converted_and_lose_area(self):
        board = Board(7, 7)
        for cell in ((2, 3), (3, 2), (3, 4), (4, 3)):
            board.set_cell(*cell, Color.RED)
        for cell in ((1, 2), (1, 3), (1, 4)):
            board.set_cell(*cell, Color.BLUE)
        sim = Simulation(board, quiet_rules(capture_enabled=True, conversion_probability=1))
        sim.capture()
        self.assertIn((2, 3), sim.protected_points())
        sim.step()
        self.assertEqual(sim.board.points[2][3], Color.BLUE)
        self.assertEqual(sim.board.owners[3][3], Color.EMPTY)


if __name__ == "__main__":
    unittest.main()
