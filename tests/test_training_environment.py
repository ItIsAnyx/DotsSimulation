import unittest
from unittest.mock import patch

from dots_simulation.agents import TinyPolicy
from dots_simulation.game import Action, ActionKind, Game
from dots_simulation.model import Board, Color, Rules, Simulation, SPECIES
from tests.test_game import solid_board
from tests.test_model import quiet_rules
from train import play_episode


class EnvironmentTests(unittest.TestCase):
    def test_neutral_placement_anywhere_costs_one_and_rewards_five(self):
        game = Game(Simulation(Board(8, 8), quiet_rules(capture_enabled=True)), automaton_rounds=0)
        game.advance()
        game.players[Color.BLUE].budget = 2
        action = Action(ActionKind.PLACE, 7, 7)
        self.assertIn(action, game.legal_actions())
        self.assertTrue(game.is_legal_action(action))
        self.assertEqual(game.apply_action(action), {Color.BLUE: 5, Color.RED: 0})
        self.assertEqual(game.players[Color.BLUE].budget, 1)
        self.assertEqual(game.simulation.board.points[7][7], Color.BLUE)
        self.assertFalse(game.is_legal_action(action))
        self.assertNotIn(action, game.legal_actions())
        # Losing and reclaiming the same neutral coordinate cannot farm rewards.
        game.simulation.board.set_cell(7, 7)
        game.touched[Color.BLUE].clear()
        self.assertEqual(game.apply_action(action), {Color.BLUE: 0, Color.RED: 0})

    def test_neutral_placement_closes_contour_and_rewards_empty_interior(self):
        board = Board(5, 5)
        for r, c in ((1, 2), (2, 1), (2, 3)):
            board.set_cell(r, c, Color.BLUE)
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)), automaton_rounds=0)
        game.advance()
        game.players[Color.BLUE].budget = 1
        rewards = game.apply_action(Action(ActionKind.PLACE, 3, 2))
        self.assertEqual(rewards[Color.BLUE], 10)  # Placed point + enclosed empty cell.
        self.assertEqual(game.simulation.board.owners[2][2], Color.BLUE)
        self.assertEqual(game.simulation.board.points[2][2], Color.EMPTY)
        game.finish_agent_turn()
        game.finish_agent_turn()
        game.advance()
        self.assertEqual(game.players[Color.BLUE].income, 1)

    def test_enemy_and_dissolved_cells_cannot_be_placed_into(self):
        board = Board(4, 4)
        board.set_cell(1, 1, Color.RED)
        board.dissolved.add((3, 3))
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)), automaton_rounds=0)
        game.advance()
        game.players[Color.BLUE].budget = 10
        for cell in ((1, 1), (3, 3)):
            action = Action(ActionKind.PLACE, *cell)
            self.assertNotIn(action, game.legal_actions())
            self.assertFalse(game.is_legal_action(action))

    def test_exactly_eight_full_phases_then_no_births_deaths_or_conversion(self):
        sim = Simulation(Board(12, 12), quiet_rules(random_birth_chance=1), seed=42)
        game = Game(sim, agents_enabled=False, round_limit=20, automaton_rounds=8)
        for _ in range(8):
            self.assertTrue(game.automaton_active)
            self.assertEqual(game.advance().automaton.births, 1)
        before = sim.board.to_ascii()
        self.assertFalse(game.automaton_active)
        for _ in range(3):
            result = game.advance()
            self.assertEqual((result.automaton.births, result.automaton.deaths,
                              result.automaton.conversions), (0, 0, 0))
            self.assertEqual(sim.board.to_ascii(), before)
        self.assertEqual(game.round, 11)
        self.assertEqual(game.clone().automaton_rounds, 8)

    def test_disabled_automaton_still_dissolves_protects_boundary_and_pays_income(self):
        sim = Simulation(solid_board(size=7, color=Color.BLUE),
                         Rules(interior_dissolve_chance=1, random_birth_chance=1,
                               growth_probability=1, isolated_death=1, conversion_probability=1), seed=42)
        game = Game(sim, automaton_rounds=0)
        result = game.advance()
        self.assertGreater(result.automaton.dissolved_points, 0)
        self.assertEqual(result.highlighted_cells, frozenset(sim.board.dissolved))
        for i in range(7):
            for r, c in ((0, i), (6, i), (i, 0), (i, 6)):
                self.assertEqual(sim.board.points[r][c], Color.BLUE)
        self.assertEqual(game.players[Color.BLUE].income, sim.board.statistics()['blue']['empty_territory'])
        self.assertEqual(game.players[Color.BLUE].budget, game.players[Color.BLUE].income)
        self.assertEqual(result.rewards[Color.BLUE], 0)

    def test_selfplay_default_disables_main_automaton_after_eight_rounds(self):
        games = []
        def create(*args, **kwargs):
            game = Game(*args, **kwargs)
            games.append(game)
            return game
        policies = {c: TinyPolicy(int(c)) for c in SPECIES}
        with patch('train.Game', side_effect=create):
            play_episode(policies, 42, 12, 12, 0, 10, quiet_rules(random_birth_chance=1), learn=False)
        self.assertEqual(games[0].automaton_rounds, 8)
        board = games[0].simulation.board
        self.assertEqual(sum(board.points[r][c] != Color.EMPTY for r, c in board.cells()), 8)
        self.assertEqual(Rules().interior_dissolve_chance, .02)


if __name__ == '__main__':
    unittest.main()
