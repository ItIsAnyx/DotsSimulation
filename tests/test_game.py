import unittest

from dots_simulation.model import Board, Color, Simulation
from dots_simulation.game import Game, Action, ActionKind, PASS
from tests.test_model import quiet_rules


def solid_board(size=5, color=Color.RED):
    board = Board(size, size)
    for r, c in board.cells():
        board.set_cell(r, c, color)
    return board


class GameTests(unittest.TestCase):
    def test_dissolve_is_irreversible_and_keeps_closed_boundary(self):
        sim = Simulation(solid_board(), quiet_rules(capture_enabled=True, interior_dissolve_chance=1))
        result = sim.step()
        self.assertGreater(result.dissolved_points, 0)
        for r, c in sim.board.dissolved:
            self.assertIn((r, c), result.highlighted_cells)
            self.assertEqual(sim.board.owners[r][c], Color.RED)
            self.assertIn((r, c), sim.enclosed_cells(Color.RED))
        for i in range(5):
            for r, c in ((0, i), (4, i), (i, 0), (i, 4)):
                self.assertEqual(sim.board.points[r][c], Color.RED)
        r, c = next(iter(sim.board.dissolved))
        with self.assertRaises(ValueError):
            sim.board.set_cell(r, c, Color.RED)
        sim.rules = quiet_rules(growth_probability=1, random_birth_chance=1)
        sim.step()
        self.assertEqual(sim.board.points[r][c], Color.EMPTY)
        self.assertIn((r, c), sim.board.copy().dissolved)

    def test_phase_order_income_and_carryover(self):
        board = solid_board(color=Color.BLUE)
        board.set_cell(2, 2, Color.EMPTY, Color.BLUE)
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
        game.advance()
        self.assertEqual(game.active_color, Color.BLUE)
        self.assertEqual(game.players[Color.BLUE].budget, 1)
        game.advance()
        self.assertEqual(game.active_color, Color.RED)
        game.advance()
        game.advance()
        self.assertEqual(game.active_color, Color.RED)
        self.assertEqual(game.players[Color.BLUE].budget, 2)

    def test_remove_only_interior_and_cost_one(self):
        game = Game(Simulation(solid_board(color=Color.BLUE), quiet_rules(capture_enabled=True)))
        game.advance()
        game.players[Color.BLUE].budget = 2
        legal = game.legal_actions()
        self.assertIn(Action(ActionKind.REMOVE, 2, 2), legal)
        self.assertNotIn(Action(ActionKind.REMOVE, 0, 2), legal)
        reward = game.apply_action(Action(ActionKind.REMOVE, 2, 2))
        self.assertEqual(reward[Color.BLUE], 0)
        self.assertEqual(game.players[Color.BLUE].budget, 1)
        self.assertEqual(game.simulation.board.owners[2][2], Color.BLUE)
        self.assertNotIn(Action(ActionKind.PLACE, 2, 2), game.legal_actions())

    def test_place_only_own_area_and_never_dissolved_cell(self):
        board = solid_board(color=Color.BLUE)
        board.set_cell(2, 2, Color.EMPTY, Color.BLUE)
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
        game.advance()
        action = Action(ActionKind.PLACE, 2, 2)
        self.assertIn(action, game.legal_actions())
        game.simulation.board.dissolved.add((2, 2))
        self.assertNotIn(action, game.legal_actions())
        with self.assertRaises(ValueError):
            game.apply_action(Action(ActionKind.REMOVE, 0, 0))

    def test_capture_cost_and_both_players_rewards(self):
        board = Board(5, 5)
        board.set_cell(2, 2, Color.RED)
        for r, c in ((1, 1), (1, 2), (1, 3)):
            board.set_cell(r, c, Color.BLUE)
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
        game.advance()
        game.players[Color.BLUE].budget = 3
        reward = game.apply_action(Action(ActionKind.CAPTURE, 2, 2))
        self.assertEqual(reward, {Color.RED: -2, Color.BLUE: 5})
        self.assertEqual(game.players[Color.BLUE].budget, 0)

    def test_first_empty_capture_and_loss_are_not_farmable(self):
        board = Board(5, 5)
        for r, c in ((1, 2), (2, 1), (2, 3), (3, 2)):
            board.set_cell(r, c, Color.BLUE)
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
        self.assertEqual(game.advance().rewards[Color.BLUE], 5)
        before = game.simulation.board.copy()
        game.simulation.board.set_cell(1, 2)
        game.simulation.capture()
        loss = game._score_changes(before)
        self.assertEqual(loss[Color.BLUE], -7)  # One point + one empty territory cell.
        before = game.simulation.board.copy()
        game.simulation.board.set_cell(1, 2, Color.BLUE)
        game.simulation.capture()
        reward = game._score_changes(before)
        self.assertEqual(reward[Color.BLUE], 0)

    def test_terminal_bonus_once_after_both_turns(self):
        board = Board(3, 3)
        board.set_cell(1, 1, Color.BLUE)
        game = Game(Simulation(board, quiet_rules()), round_limit=1)
        game.advance()
        self.assertFalse(game.done)
        game.advance()
        result = game.advance()
        self.assertTrue(game.done)
        self.assertEqual(game.winner, Color.BLUE)
        self.assertEqual(result.rewards[Color.BLUE], 50)
        with self.assertRaises(ValueError):
            game.advance()
        self.assertEqual(game.players[Color.BLUE].score, 50)

    def test_disabled_agents_preserve_autonomous_phase(self):
        game = Game(Simulation(Board(3, 3), quiet_rules()), agents_enabled=False, round_limit=2)
        game.advance()
        self.assertIsNone(game.active_color)
        game.advance()
        self.assertTrue(game.done)
        self.assertEqual(game.winner, Color.EMPTY)


if __name__ == "__main__":
    unittest.main()
