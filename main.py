"""Run the GUI in PyCharm; use --console for ASCII output."""

import argparse
from dots_simulation import Rules, Simulation


def main():
    parser = argparse.ArgumentParser(description="Two-color dots cellular automaton")
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--height", type=int, default=30)
    parser.add_argument("--density", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--random-birth-chance", type=float, default=0.2,
                        help="Probability of one random point birth per turn (0..1)")
    parser.add_argument("--dissolve-chance", type=float, default=0.0005)
    parser.add_argument("--round-limit", type=int, default=200)
    parser.add_argument("--agents", choices=("off", "random", "heuristic", "neural", "search"), default="heuristic")
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--search-beam", type=int, default=4)
    parser.add_argument("--search-branches", type=int, default=5)
    parser.add_argument("--search-nodes", type=int, default=100)
    parser.add_argument("--model", help="JSON checkpoint of two trained policies")
    parser.add_argument("--console", action="store_true", help="Print ASCII simulation instead of opening a window")
    args = parser.parse_args()
    from dots_simulation.search import SearchAgent, SearchConfig
    try:
        search_config = SearchConfig(args.search_depth, args.search_beam, args.search_branches, args.search_nodes)
    except ValueError as error:
        parser.error(str(error))
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    if not 0 <= args.random_birth_chance <= 1:
        parser.error("--random-birth-chance must be between 0 and 1")
    if not 0 <= args.dissolve_chance <= 1 or args.round_limit < 1:
        parser.error("Invalid dissolve chance or round limit")
    simulation = Simulation.random_field(args.width, args.height, args.density,
                                        rules=Rules(random_birth_chance=args.random_birth_chance,
                                                    interior_dissolve_chance=args.dissolve_chance), seed=args.seed)
    if not args.console:
        from dots_simulation.gui import launch
        modes = dict(off="Выключены", random="Случайные", heuristic="Эвристика", neural="Нейросеть", search="Нейросеть + поиск")
        launch(simulation, args.density, args.seed, args.round_limit, modes[args.agents], args.model, search_config)
        return
    print("R/B: points; r/b: empty owned territory; .: free cell")
    print("Generation 0")
    print(simulation.board.to_ascii())
    from dots_simulation.game import Game
    from dots_simulation.agents import HeuristicAgent, NeuralAgent, RandomAgent, load_policies
    game = Game(simulation, agents_enabled=args.agents != "off", round_limit=args.round_limit)
    from dots_simulation.model import SPECIES
    if args.model:
        policies = load_policies(args.model)
        agents = {c: SearchAgent(policies[c], seed=int(c), config=search_config) if args.agents == "search"
                  else NeuralAgent(policies[c]) for c in SPECIES}
    elif args.agents == "search":
        agents = {c: SearchAgent(seed=int(c), config=search_config) for c in SPECIES}
    else:
        factory = {"off": HeuristicAgent, "heuristic": HeuristicAgent, "random": RandomAgent, "neural": NeuralAgent}[args.agents]
        agents = {c: factory() for c in SPECIES}
    for _ in range(args.steps):
        if game.done:
            break
        result = game.advance(agents)
        print(f"\n{result}")
        print(simulation.board.to_ascii())
    print("\nStatistics:", simulation.board.statistics())
    print("Players:", game.players)


if __name__ == "__main__":
    main()
