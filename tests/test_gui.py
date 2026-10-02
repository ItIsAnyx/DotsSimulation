"""GUI integration smoke test; skipped on Python installations without Tcl/Tk."""

import unittest

try:
    import tkinter as tk
    from dots_simulation.gui import SimulationApp
except ImportError:
    tk = None

from dots_simulation import Board, Color, Rules, Simulation


@unittest.skipIf(tk is None, "Tkinter is not installed")
class GuiTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tcl/Tk is unavailable: {error}")
        self.addCleanup(self.root.destroy)
        self.root.withdraw()

    def test_search_mode_uses_search_agents_and_preserves_loaded_weights(self):
        from dots_simulation.agents import TinyPolicy
        from dots_simulation.search import SearchAgent
        app = SimulationApp(self.root, Simulation.random_field(10, 8, seed=17))
        app.policies = {c: TinyPolicy(int(c)) for c in (Color.RED, Color.BLUE)}
        app.agent_mode.set("Нейросеть + поиск")
        app._change_agents()
        self.assertTrue(all(isinstance(agent, SearchAgent) for agent in app.agents.values()))
        self.assertIn("Поиск: глубина 4", app.agent_status.get())
        app.game.advance()
        app.agents[Color.BLUE].choose(app.game, app.game.legal_actions())
        self.assertEqual(app.policies[Color.BLUE].as_dict(), app.agents[Color.BLUE].policy.as_dict())
        app.agent_mode.set("Нейросеть")
        app._change_agents()
        self.assertFalse(any(isinstance(agent, SearchAgent) for agent in app.agents.values()))

    def test_render_restart_and_timer_lifecycle(self):
        app = SimulationApp(self.root, Simulation.random_field(10, 8, seed=17))
        app.agent_mode.set("Выключены")
        app._change_agents()
        # Fix canvas dimensions for a hidden-window rendering check.
        app.canvas.configure(width=600, height=480)
        self.root.update_idletasks()
        initial = app.simulation.board.to_ascii()
        app.single_step()
        first_step = app.simulation.board.to_ascii()
        self.assertEqual(app.simulation.generation, 1)
        app.restart()
        self.assertEqual(app.simulation.board.to_ascii(), initial)
        app.single_step()
        self.assertEqual(app.simulation.board.to_ascii(), first_step)
        app.toggle()
        self.assertIsNotNone(app.timer)
        app.pause()
        self.assertIsNone(app.timer)
        app.width.set("12")
        app.height.set("9")
        app.seed.set("18")
        app.new_field()
        self.assertEqual((app.simulation.board.width, app.simulation.board.height), (12, 9))
        self.assertEqual(app.simulation.generation, 0)
        self.assertFalse(app.running)
        app.canvas.winfo_width = lambda: 600
        app.canvas.winfo_height = lambda: 480
        app._draw()
        self.assertEqual(len(app.rectangles), 108)
        self.assertEqual(len(app.dots), 108)
        for item, (r, c) in zip(app.dots, app.simulation.board.cells()):
            expected = "normal" if app.simulation.board.points[r][c] else "hidden"
            self.assertEqual(app.canvas.itemcget(item, "state"), expected)
        app.pause()

    def test_live_birth_chance_and_capture_colors(self):
        from dots_simulation.gui import NEW_POINTS, NEW_TERRITORY, POINTS, TERRITORY
        board = Board(5, 5)
        for cell in ((1, 2), (2, 1), (2, 3), (3, 2)):
            board.set_cell(*cell, Color.RED)
        rules = Rules(random_birth_chance=0, growth_probability=0,
                      isolated_death=0, conversion_probability=0)
        app = SimulationApp(self.root, Simulation(board, rules))
        app.agent_mode.set("Выключены")
        app._change_agents()
        app.canvas.winfo_width = lambda: 600
        app.canvas.winfo_height = lambda: 480
        app.single_step()
        center = 2 * 5 + 2
        self.assertEqual(app.canvas.itemcget(app.rectangles[center], "fill"), NEW_TERRITORY[Color.RED])
        app.single_step()
        self.assertEqual(app.canvas.itemcget(app.rectangles[center], "fill"), TERRITORY[Color.RED])
        board.set_cell(2, 2, Color.BLUE)
        app.simulation = Simulation(board, rules)
        app.single_step()
        self.assertEqual(app.canvas.itemcget(app.dots[center], "fill"), NEW_POINTS[Color.RED])
        app.single_step()
        self.assertEqual(app.canvas.itemcget(app.dots[center], "fill"), POINTS[Color.RED])
        app.birth_chance.set("1")
        app.apply_birth_chance()
        self.assertEqual(app.simulation.rules.random_birth_chance, 1)
        app.birth_chance.set("0,25")
        app.new_field()
        self.assertEqual(app.simulation.rules.random_birth_chance, 0.25)
        self.assertIsNone(app.last_result)
        app.pause()

    def test_three_phases_budget_panels_and_async_turn(self):
        from tests.test_game import solid_board
        from tests.test_model import quiet_rules
        board = solid_board(color=Color.BLUE)
        board.set_cell(2, 2, Color.EMPTY, Color.BLUE)
        app = SimulationApp(self.root, Simulation(board, quiet_rules(capture_enabled=True)))
        self.addCleanup(app._cancel_phase)
        app.single_step()
        self.assertEqual(app.game.active_color, Color.BLUE)
        self.assertEqual(app.game.players[Color.BLUE].budget, 1)
        self.assertIn("Действий: 1", app.blue_statistics.get())
        app.single_step()
        self.assertTrue(app.phase_busy)
        self.root.after(100, self.root.quit)
        self.root.mainloop()
        self.assertFalse(app.phase_busy)
        self.assertEqual(app.game.active_color, Color.RED)
        self.assertEqual(app.simulation.generation, 1)
        app.single_step()
        self.root.after(100, self.root.quit)
        self.root.mainloop()
        self.assertEqual(app.game.next_phase, "Автомат")
        app.single_step()
        self.assertEqual(app.game.active_color, Color.RED)
        app.pause()

    def test_statistics_deltas_cover_whole_phase_and_reset(self):
        from tests.test_game import solid_board
        from tests.test_model import quiet_rules
        board = solid_board(color=Color.BLUE)
        board.set_cell(2, 2, Color.EMPTY, Color.BLUE)
        app = SimulationApp(self.root, Simulation(board, quiet_rules(capture_enabled=True)))
        self.addCleanup(app._cancel_phase)
        self.assertIn("Точки: 24 (+0)", app.blue_statistics.get())
        app.single_step()  # Automaton leaves this prepared field unchanged.
        self.assertIn("Территория: 25 (+0)", app.blue_statistics.get())
        app.single_step()  # Blue removes one interior point with its one resource.
        self.root.after(100, self.root.quit)
        self.root.mainloop()
        self.assertIn("Точки: 23 (-1)", app.blue_statistics.get())
        self.assertIn("Территория: 25 (+0)", app.blue_statistics.get())
        self.assertIn("Без точек: 2 (+1)", app.blue_statistics.get())
        app.refresh()  # Repainting / pausing must not erase phase deltas.
        self.assertIn("Точки: 23 (-1)", app.blue_statistics.get())
        app.single_step()  # Red forced PASS: new phase, all deltas now zero.
        self.root.after(100, self.root.quit)
        self.root.mainloop()
        self.assertIn("Точки: 23 (+0)", app.blue_statistics.get())
        app.restart()
        self.assertIn("Точки: 24 (+0)", app.blue_statistics.get())
        app.width.set("6")
        app.height.set("6")
        app.new_field()
        for color, var in ((Color.RED, app.red_statistics), (Color.BLUE, app.blue_statistics)):
            stats = app.simulation.board.statistics()[color.name.lower()]
            self.assertIn(f"Точки: {stats['points']} (+0)", var.get())
        app.pause()


if __name__ == "__main__":
    unittest.main()
