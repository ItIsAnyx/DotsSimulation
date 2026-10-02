"""Offline REINFORCE self-play; no GUI or third-party libraries required."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from dots_simulation.agents import TinyPolicy, HeuristicAgent, NeuralAgent, save_policies, load_policies
from dots_simulation.game import Game, ActionKind
from dots_simulation.model import Color, Rules, Simulation, SPECIES


def play_episode(policies, seed, width, height, density, rounds, rules, learn=True,
                 learning_rate=0.01, gamma=0.995, initial_board=None, active=False):
    simulation = (Simulation(initial_board, rules, seed) if initial_board is not None else
                  Simulation.random_field(width, height, density, rules=rules, seed=seed))
    game = Game(simulation, round_limit=rounds)
    trajectories = {color: [] for color in SPECIES}
    decisions = {color: 0 for color in SPECIES}
    moves = {kind.value: 0 for kind in ActionKind}
    stats = dict(early_passes=0, captures_left_on_pass=0, spent_actions=0,
                 red_learning_return=0.0, blue_learning_return=0.0,
                 red_final_budget=0, blue_final_budget=0)

    def credit(rewards):
        # Include opponent and automaton consequences until the next decision.
        # Before a player's first decision there is no causal action to reward.
        for color in SPECIES:
            enemy = Color.BLUE if color == Color.RED else Color.RED
            learning_reward = rewards[color] - rewards[enemy]
            # Zero-sum learning signal cancels profitable point-trading loops.
            # UI/game scores retain the user's +5/-2 event accounting.
            stats[f"{color.name.lower()}_learning_return"] += learning_reward
            if trajectories[color]:
                trajectories[color][-1].reward += learning_reward

    while not game.done:
        if game.phase_index == 0:
            credit(game.advance_automaton().rewards)
            continue
        color = game.active_color
        while True:
            legal = game.legal_actions()
            action, decision = policies[color].decide(game, legal, active=active)
            if action.kind == ActionKind.PASS:
                captures = sum(a.kind == ActionKind.CAPTURE for a in legal)
                stats["early_passes"] += captures > 0
                stats["captures_left_on_pass"] += captures
            policies[color].cache_gradient(decision)
            trajectories[color].append(decision)
            decisions[color] += decision.choice_count > 1
            credit(game.apply_action(action))
            stats["spent_actions"] += action.cost
            moves[action.kind.value] += 1
            if action.kind == ActionKind.PASS:
                break
        credit(game.finish_agent_turn())
    norms = {c: policies[c].update(trajectories[c], learning_rate, gamma) if learn else 0 for c in SPECIES}
    stats["red_final_budget"] = game.players[Color.RED].budget
    stats["blue_final_budget"] = game.players[Color.BLUE].budget
    return dict(winner=game.winner.name.lower(), red_score=game.players[Color.RED].score,
                blue_score=game.players[Color.BLUE].score,
                red_decisions=decisions[Color.RED], blue_decisions=decisions[Color.BLUE],
                moves=sum(v for k, v in moves.items() if k != "pass"),
                place=moves["place"], remove=moves["remove"], capture=moves["capture"],
                red_gradient_norm=norms[Color.RED], blue_gradient_norm=norms[Color.BLUE], **stats)


def evaluate(policies, seeds, width, height, density, rounds, rules, active=False):
    """Freeze weights, play each color against the same inspectable baseline."""
    results = dict(wins=0, losses=0, draws=0, games=0)
    for seed in seeds:
        for color in SPECIES:
            enemy = Color.BLUE if color == Color.RED else Color.RED
            game = Game(Simulation.random_field(width, height, density, rules=rules, seed=seed), round_limit=rounds)
            clone = TinyPolicy.from_dict(policies[color].as_dict(), seed + int(color))
            agents = {color: NeuralAgent(clone, active=active), enemy: HeuristicAgent()}
            while not game.done:
                game.advance(agents)
            results["games"] += 1
            key = "draws" if game.winner == Color.EMPTY else "wins" if game.winner == color else "losses"
            results[key] += 1
    return results


def main():
    parser = argparse.ArgumentParser(description="Train two small policies through self-play")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--density", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount per game round, not per action")
    parser.add_argument("--allow-early-pass", action="store_true", help="Compatibility flag: early PASS is already allowed by default")
    parser.add_argument("--legacy-active", action="store_true", help="Explicitly enable the old hand-written capture/PASS guard")
    parser.add_argument("--dissolve-chance", type=float, default=0.0005)
    parser.add_argument("--random-birth-chance", type=float, default=0.2)
    parser.add_argument("--output", default="models/selfplay.json")
    parser.add_argument("--resume", help="Load weights and running reward baselines")
    parser.add_argument("--eval-games", type=int, default=4, help="Number of held-out seeds (two colors per seed)")
    args = parser.parse_args()
    if (min(args.episodes, args.rounds, args.width, args.height) < 1 or args.eval_games < 0
            or not 0 < args.learning_rate <= 1 or not 0 <= args.gamma <= 1
            or not 0 <= args.density <= 1):
        parser.error("Invalid training parameters")
    try:
        rules = Rules(random_birth_chance=args.random_birth_chance,
                      interior_dissolve_chance=args.dissolve_chance)
    except ValueError as error:
        parser.error(str(error))
    policies = load_policies(args.resume, args.seed) if args.resume else {c: TinyPolicy(args.seed + int(c)) for c in SPECIES}
    episode_base = min(policy.episodes for policy in policies.values())
    evaluation_seeds = range(args.seed + 1_000_000, args.seed + 1_000_000 + args.eval_games)
    before = evaluate(policies, evaluation_seeds, args.width, args.height, args.density, args.rounds, rules,
                      active=args.legacy_active and not args.allow_early_pass)
    print("Before training vs heuristic:", before, flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = destination.with_suffix(".csv")
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for episode in range(args.episodes):
            result = play_episode(policies, args.seed + episode_base + episode, args.width, args.height,
                                  args.density, args.rounds, rules, learning_rate=args.learning_rate, gamma=args.gamma,
                                  active=args.legacy_active and not args.allow_early_pass)
            row = dict(episode=episode_base + episode + 1, seed=args.seed + episode_base + episode, **result)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"Episode {episode + 1}/{args.episodes}: {result}", flush=True)
            # Checkpoint each completed episode so an interrupted run keeps progress.
            save_policies(destination, policies, dict(config=vars(args), rules=asdict(rules), evaluation_before=before))
    after = evaluate(policies, evaluation_seeds, args.width, args.height, args.density, args.rounds, rules,
                     active=args.legacy_active and not args.allow_early_pass)
    save_policies(destination, policies, dict(config=vars(args), rules=asdict(rules),
                                            evaluation_before=before, evaluation_after=after))
    print("After training vs heuristic:", after)
    print(f"Weights: {destination.resolve()}\nLog: {log_path.resolve()}")


if __name__ == "__main__":
    main()
