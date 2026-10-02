import copy
import unittest
from unittest.mock import patch

from dots_simulation.agents import (FEATURE_NAMES, LEGACY_FEATURE_NAMES, TinyPolicy,
                                    action_features, state_features)
from dots_simulation.game import Action, ActionKind, Game, PASS
from dots_simulation.model import Board, Color, Simulation, SPECIES
from dots_simulation.search import SearchAgent, SearchConfig
from tests.test_model import quiet_rules
from tests.test_activity import capture_game
from train import play_episode


def chain_game():
    board = Board(6, 6)
    for r, c in ((1, 1), (1, 2), (1, 3), (1, 4), (3, 1)):
        board.set_cell(r, c, Color.BLUE)
    for r, c in ((2, 2), (2, 3), (3, 2)):
        board.set_cell(r, c, Color.RED)
    game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
    game.advance()
    game.players[Color.BLUE].budget = 9
    return game


class SearchTests(unittest.TestCase):
    def test_cloned_game_has_independent_board_budget_history_and_rng(self):
        game = capture_game()
        clone = game.clone()
        self.assertEqual(clone.simulation.random.getstate(), game.simulation.random.getstate())
        clone.apply_action(next(a for a in clone.legal_actions() if a != PASS))
        clone.simulation.board.dissolved.add((0, 0))
        clone.rewarded_empty.add((0, 0))
        clone.simulation.random.random()
        self.assertEqual(game.players[Color.BLUE].budget, 100)
        self.assertEqual(game.touched[Color.BLUE], set())
        self.assertNotIn((0, 0), game.simulation.board.dissolved)
        self.assertNotIn((0, 0), game.rewarded_empty)
        self.assertNotEqual(clone.simulation.random.getstate(), game.simulation.random.getstate())
        self.assertNotEqual(clone.simulation.board.to_ascii(), game.simulation.board.to_ascii())

    def test_search_finds_continuation_unavailable_before_first_two_actions(self):
        game = chain_game()
        third = Action(ActionKind.CAPTURE, 3, 2)
        self.assertNotIn(third, game.legal_actions())
        agent = SearchAgent(TinyPolicy(seed=2), config=SearchConfig(depth=3, beam=4, branches=5))
        before = game.simulation.board.to_ascii()
        weights = copy.deepcopy(agent.policy.as_dict())
        rng = agent.policy.random.getstate()
        choice = agent.choose(game, game.legal_actions())
        self.assertEqual(len(agent.last_sequence), 3)
        self.assertIn(third, agent.last_sequence)
        self.assertEqual(choice, agent.last_sequence[0])
        clone = game.clone()
        for action in agent.last_sequence:
            self.assertTrue(clone.is_legal_action(action))
            clone.apply_action(action)
        self.assertEqual(clone.players[Color.BLUE].budget, 0)
        self.assertEqual(game.simulation.board.to_ascii(), before)
        self.assertEqual(game.players[Color.BLUE].budget, 9)
        self.assertEqual(agent.policy.as_dict(), weights)
        self.assertEqual(agent.policy.random.getstate(), rng)

    def test_search_limits_and_seed_reproducibility(self):
        game = chain_game()
        config = SearchConfig(depth=5, max_nodes=7)
        agents = [SearchAgent(TinyPolicy(seed=2), seed=4, config=config) for _ in range(2)]
        self.assertEqual(agents[0].choose(game, game.legal_actions()), agents[1].choose(game, game.legal_actions()))
        self.assertEqual(agents[0].last_sequence, agents[1].last_sequence)
        self.assertLessEqual(agents[0].last_nodes, 7)
        game.players[Color.BLUE].budget = 0
        self.assertEqual(agents[0].choose(game, game.legal_actions()), PASS)
        with self.assertRaises(ValueError):
            SearchConfig(depth=0)

    def test_observation_distinguishes_geometry_and_distant_changes(self):
        game = chain_game()
        action = Action(ActionKind.CAPTURE, 2, 2)
        first = action_features(game, [action])[0]
        self.assertEqual(len(first), len(FEATURE_NAMES))
        game.simulation.board.set_cell(1, 1, Color.EMPTY)
        game.simulation.board.set_cell(2, 1, Color.BLUE)
        second = action_features(game, [action])[0]
        self.assertEqual(first[4:9], second[4:9])  # Equal counts, different shape.
        self.assertNotEqual(first[20:263], second[20:263])
        board = Board(20, 20)
        distant = Game(Simulation(board, quiet_rules()))
        distant.advance()
        first = state_features(distant, Color.BLUE)
        distant.simulation.board.set_cell(0, 0, Color.RED)
        second = state_features(distant, Color.BLUE)
        self.assertEqual(first[20:263], second[20:263])
        self.assertNotEqual(first[263:327], second[263:327])

    def test_value_learns_returns_and_old_actor_migrates_with_zero_new_weights(self):
        policy = TinyPolicy(seed=2)
        observation = state_features(chain_game(), Color.BLUE)
        before = policy.value.predict(observation)
        policy.value.fit([(observation, 5.0)], 0.1)
        self.assertGreater(policy.value.predict(observation), before)
        self.assertEqual(TinyPolicy.from_dict(policy.as_dict()).as_dict(), policy.as_dict())
        old = copy.deepcopy(policy.as_dict())
        old['features'] = list(LEGACY_FEATURE_NAMES)
        old['w1'] = [row[:20] for row in old['w1']]
        old['training_version'] = 2
        old.pop('value')
        restored = TinyPolicy.from_dict(old)
        self.assertTrue(all(row[20:] == [0.0] * (len(FEATURE_NAMES) - 20) for row in restored.w1))
        self.assertEqual(restored.value.predict(observation), 0)

    def test_training_and_default_policy_do_not_force_search_or_captures(self):
        game = chain_game()
        policy = TinyPolicy()
        self.assertIn(PASS, policy.distribution(game, game.legal_actions())[0])
        policies = {c: TinyPolicy(int(c)) for c in SPECIES}
        with patch.object(SearchAgent, 'choose', side_effect=AssertionError('Search in training')):
            play_episode(policies, 3, 6, 6, 0.1, 3, quiet_rules(capture_enabled=True),
                         initial_board=game.simulation.board)

    def test_value_gradient_matches_finite_differences(self):
        value = TinyPolicy(seed=3, hidden=3).value
        value.w2 = [0.05, -0.03, 0.07]
        x = state_features(chain_game(), Color.BLUE)
        target = 0.1
        parameters = [(value.w1[1], 273), (value.b1, 1), (value.w2, 1)]
        expected = []
        for container, index in parameters:
            original, epsilon = container[index], 1e-6
            container[index] = original + epsilon
            upper = 0.5 * (value.predict(x) - target) ** 2
            container[index] = original - epsilon
            lower = 0.5 * (value.predict(x) - target) ** 2
            container[index] = original
            expected.append((upper - lower) / (2 * epsilon))
        originals = [container[index] for container, index in parameters]
        value.fit([(x, target)], learning_rate=1e-5)
        for (container, index), original, gradient in zip(parameters, originals, expected):
            self.assertAlmostEqual((original - container[index]) / 1e-5, gradient, places=7)


if __name__ == '__main__':
    unittest.main()
