import unittest

from dots_simulation import Board, Color, Simulation
from dots_simulation.agents import Decision, TinyPolicy, NeuralAgent
from dots_simulation.game import ActionKind, Game, PASS
from tests.test_model import quiet_rules
from tests.test_game import solid_board


def capture_game():
    board = Board(6, 5)
    for r, c in ((1, 1), (1, 2), (1, 3), (1, 4)):
        board.set_cell(r, c, Color.BLUE)
    for r, c in ((2, 2), (2, 3)):
        board.set_cell(r, c, Color.RED)
    game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
    game.advance()
    game.players[Color.BLUE].budget = 100
    return game


class ActivityTests(unittest.TestCase):
    def test_neural_turn_uses_all_available_captures(self):
        game = capture_game()
        policy = TinyPolicy(seed=9)
        # Even a network strongly preferring PASS cannot skip legal captures.
        policy.w1 = [[0.0] * 20 for _ in range(16)]
        policy.w1[0][0] = 100
        policy.w2 = [100] + [0] * 15
        result = game.advance({Color.BLUE: NeuralAgent(policy, active=True)})
        self.assertEqual(result.action_count, 2)
        self.assertEqual(game.players[Color.BLUE].spent, 6)
        self.assertEqual(game.players[Color.BLUE].budget, 94)
        self.assertEqual(game.simulation.board.points[2][2], Color.BLUE)
        self.assertEqual(game.simulation.board.points[2][3], Color.BLUE)

    def test_guard_is_explicit_and_optional(self):
        game, policy = capture_game(), TinyPolicy()
        legal = game.legal_actions()
        candidates, _, _, probabilities = policy.distribution(game, legal, active=True)
        self.assertNotIn(PASS, candidates)
        self.assertAlmostEqual(sum(probabilities), 1)
        self.assertIn(PASS, policy.distribution(game, legal, active=False)[0])
        game.players[Color.BLUE].budget = 2
        self.assertIn(PASS, policy.distribution(game, game.legal_actions())[0])

    def test_budget_feature_does_not_saturate_at_twenty(self):
        from dots_simulation.agents import action_features
        game = capture_game()
        game.players[Color.BLUE].budget = 100
        small = action_features(game, [PASS])[0][14]
        game.players[Color.BLUE].budget = 4000
        large = action_features(game, [PASS])[0][14]
        self.assertGreater(large, small + 0.2)

    def test_victory_and_defeat_scale_with_preterminal_scores(self):
        for blue_score, red_score in ((1000, 2000), (-1000, 2000), (1000, -2000)):
            game = Game(Simulation(solid_board(color=Color.BLUE), quiet_rules()), round_limit=1)
            game.players[Color.BLUE].score = blue_score
            game.players[Color.RED].score = red_score
            game.advance()
            game.advance()
            rewards = game.advance().rewards
            self.assertEqual(rewards[Color.BLUE], 2 * abs(blue_score) + 50)
            self.assertEqual(rewards[Color.RED], -(2 * abs(red_score) + 50))
            self.assertGreater(game.players[Color.BLUE].score, 0)
            self.assertLess(game.players[Color.RED].score, 0)
            self.assertEqual(game._finish(), {Color.RED: 0, Color.BLUE: 0})

    def test_discount_depends_on_rounds_not_action_count(self):
        features = [[1.0] + [0.0] * 19, [0.0, 1.0] + [0.0] * 18]
        same_round = [Decision(features, [0, 0], 0, round=3),
                      Decision(features, [0, 0], 1, reward=40, round=3)]
        policy = TinyPolicy()
        policy.update(same_round, learning_rate=0, gamma=0.5)
        self.assertAlmostEqual(policy.baseline, 0.1)  # Both returns = 2.
        policy = TinyPolicy()
        different_rounds = [Decision(features, [0, 0], 0, round=1),
                            Decision(features, [0, 0], 1, reward=40, round=3)]
        policy.update(different_rounds, learning_rate=0, gamma=0.5)
        self.assertAlmostEqual(policy.baseline, 0.0625)  # Returns = 0.5 and 2.

    def test_cached_gradient_matches_uncached_update(self):
        features = [[1.0] + [0.0] * 19, [0.0, 1.0] + [0.0] * 18]
        cached, plain = TinyPolicy(seed=9), TinyPolicy(seed=9)
        decision = Decision(features, [0, 0], 1, reward=40, round=1)
        cached.cache_gradient(decision)
        self.assertEqual(decision.features, [])
        self.assertEqual(decision.choice_count, 2)
        cached.update([decision])
        plain.update([Decision(features, [0, 0], 1, reward=40, round=1)])
        self.assertEqual(cached.as_dict(), plain.as_dict())

    def test_shortlist_keeps_action_types_and_never_illegal_actions(self):
        board = solid_board(size=12, color=Color.BLUE)
        board.set_cell(6, 6, Color.EMPTY, Color.BLUE)
        game = Game(Simulation(board, quiet_rules(capture_enabled=True)))
        game.advance()
        game.players[Color.BLUE].budget = 100
        legal = game.legal_actions()
        candidates = TinyPolicy().distribution(game, legal, candidate_limit=4)[0]
        self.assertLessEqual(sum(a.kind == ActionKind.REMOVE for a in candidates), 4)
        self.assertIn(PASS, candidates)
        self.assertTrue(all(a in legal and game.is_legal_action(a) for a in candidates))

    def test_legacy_checkpoint_resets_incompatible_baseline(self):
        data = TinyPolicy().as_dict()
        data.pop('training_version')
        data['baseline'] = 100
        loaded = TinyPolicy.from_dict(data)
        self.assertEqual(loaded.training_version, 1)
        self.assertEqual(loaded.baseline, 0)


if __name__ == '__main__':
    unittest.main()
