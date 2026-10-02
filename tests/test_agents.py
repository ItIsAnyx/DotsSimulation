import math
from pathlib import Path
import tempfile
import unittest

from dots_simulation.agents import TinyPolicy, Decision, FEATURE_NAMES, save_policies, load_policies
from dots_simulation.game import Game
from dots_simulation.model import Color, SPECIES, Simulation
from tests.test_model import quiet_rules
from tests.test_game import solid_board
from train import play_episode


class PolicyTests(unittest.TestCase):
    def test_masked_choice_and_color_relative_feature_count(self):
        game = Game(Simulation(solid_board(color=Color.BLUE), quiet_rules(capture_enabled=True)))
        game.advance()
        game.players[Color.BLUE].budget = 2
        legal = game.legal_actions()
        action, decision = TinyPolicy().decide(game, legal)
        self.assertIn(action, legal)
        self.assertTrue(all(len(x) == len(FEATURE_NAMES) for x in decision.features))

    def test_log_policy_gradient_matches_finite_differences(self):
        policy = TinyPolicy(seed=1, hidden=3)
        features = [[(i + j) / len(FEATURE_NAMES) for i in range(len(FEATURE_NAMES))] for j in range(3)]
        decision = Decision(features, [0, -0.5, -0.5], 1)
        gw1, gb1, gw2 = policy.gradient(decision)
        def numerical(container, index):
            old, epsilon = container[index], 1e-6
            container[index] = old + epsilon
            upper = math.log(policy.forward(features, decision.offsets)[0][1])
            container[index] = old - epsilon
            lower = math.log(policy.forward(features, decision.offsets)[0][1])
            container[index] = old
            return (upper - lower) / (2 * epsilon)
        self.assertAlmostEqual(gw1[1][7], numerical(policy.w1[1], 7), places=7)
        self.assertAlmostEqual(gw1[1][273], numerical(policy.w1[1], 273), places=7)
        self.assertAlmostEqual(gb1[1], numerical(policy.b1, 1), places=7)
        self.assertAlmostEqual(gw2[1], numerical(policy.w2, 1), places=7)

    def test_positive_reward_increases_selected_probability(self):
        policy = TinyPolicy(seed=4)
        features = [[1.0 if i == j else 0.0 for i in range(20)] for j in range(2)]
        decision = Decision(features, [0, 0], 1, reward=20)
        before = policy.forward(features)[0][1]
        policy.update([decision], learning_rate=0.1)
        self.assertGreater(policy.forward(features)[0][1], before)

    def test_checkpoint_round_trip(self):
        policies = {c: TinyPolicy(int(c)) for c in SPECIES}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            save_policies(path, policies)
            restored = load_policies(path)
            for c in SPECIES:
                self.assertEqual(restored[c].as_dict(), policies[c].as_dict())

    def test_self_play_updates_weights_and_reproduces_seed(self):
        rules = quiet_rules(capture_enabled=True, interior_dissolve_chance=0.15)
        first = {c: TinyPolicy(int(c)) for c in SPECIES}
        second = {c: TinyPolicy(int(c)) for c in SPECIES}
        before = first[Color.BLUE].as_dict()["w1"]
        before = [row[:] for row in before]
        board = solid_board(size=7, color=Color.BLUE)
        result1 = play_episode(first, 8, 7, 7, 1, 5, rules, initial_board=board)
        result2 = play_episode(second, 8, 7, 7, 1, 5, rules, initial_board=board)
        self.assertEqual(result1, result2)
        self.assertEqual(first[Color.BLUE].as_dict(), second[Color.BLUE].as_dict())
        self.assertGreater(result1["red_decisions"] + result1["blue_decisions"], 0)
        changed = before != first[Color.BLUE].w1
        self.assertTrue(changed or first[Color.RED].baseline != 0)


if __name__ == "__main__":
    unittest.main()
