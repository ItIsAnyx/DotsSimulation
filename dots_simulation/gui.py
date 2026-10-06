"""Tkinter view of the automaton; the simulation core has no UI dependencies."""

import tkinter as tk
from dataclasses import replace
from tkinter import filedialog, messagebox, ttk
from random import SystemRandom
from time import perf_counter

from .model import Color, Simulation
from .game import Game, ActionKind, PhaseResult
from .agents import HeuristicAgent, NeuralAgent, RandomAgent, TinyPolicy, load_policies
from .search import SearchAgent, SearchConfig


BACKGROUND = "#f8fafc"
TERRITORY = {Color.EMPTY: BACKGROUND, Color.RED: "#fce0df", Color.BLUE: "#dceafa"}
POINTS = {Color.RED: "#c83939", Color.BLUE: "#276bc2"}
NEW_TERRITORY = {Color.RED: "#f49b94", Color.BLUE: "#8bbcf5"}
NEW_POINTS = {Color.RED: "#ff7465", Color.BLUE: "#55a9ff"}


class SimulationApp:
    def __init__(self, root: tk.Tk, simulation: Simulation):
        self.root = root
        self.simulation = simulation
        self.round_limit = 200
        self.automaton_rounds = tk.DoubleVar(value=8)
        self.automaton_unlimited = tk.BooleanVar(value=True)
        self.automaton_limit_text = tk.StringVar(value="Без ограничения")
        self.agent_mode = tk.StringVar(value="Эвристика")
        self.policies = None
        self.search_config = SearchConfig()
        self.agent_status = tk.StringVar(value="Эвристика: правила выбора заданы вручную, обучения нет.")
        self.game = Game(simulation, round_limit=self.round_limit)
        self._reset_agents()
        self.phase_busy = False
        self.phase_timer = None
        self.initial_board = simulation.board.copy()
        self.initial_random_state = simulation.random.getstate()
        self.running = False
        self.timer = None
        self.render_timer = None
        self.rectangles = []
        self.dots = []
        self.geometry = None
        self.last_result = None
        self.phase_statistics_before = simulation.board.statistics()
        self.step_ms = 0.0
        self.speed = tk.DoubleVar(value=5)
        self.width = tk.StringVar(value=str(simulation.board.width))
        self.height = tk.StringVar(value=str(simulation.board.height))
        self.density = tk.StringVar(value="0.02")
        self.seed = tk.StringVar(value="42")
        self.birth_chance = tk.StringVar(value=str(simulation.rules.random_birth_chance))
        self.dissolve_chance = tk.StringVar(value=str(simulation.rules.interior_dissolve_chance))
        self.blue_statistics = tk.StringVar()
        self.red_statistics = tk.StringVar()
        self.grid = tk.BooleanVar(value=True)
        self.generation = tk.StringVar()
        self.statistics = tk.StringVar()
        self.events = tk.StringVar()
        self.status = tk.StringVar(value="Пауза. Пробел — запуск / пауза, → — один шаг.")
        self.hover = tk.StringVar(value="Наведите курсор на клетку, чтобы увидеть её состояние.")
        root.title("DotsSimulation — клеточный автомат")
        root.geometry("1200x850")
        root.minsize(900, 650)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        root.bind("<space>", self._keyboard_toggle)
        root.bind("<Right>", self._keyboard_step)
        self.refresh()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(5, weight=1)
        toolbar = ttk.Frame(self.root, padding=(12, 10))
        toolbar.grid(row=0, column=0, sticky="ew")
        self.run_button = ttk.Button(toolbar, text="Запуск", command=self.toggle)
        self.run_button.pack(side="left")
        self.step_button = ttk.Button(toolbar, text="Фаза →", command=self.single_step)
        self.step_button.pack(side="left", padx=6)
        ttk.Button(toolbar, text="Сначала", command=self.restart).pack(side="left")
        ttk.Label(toolbar, text="Скорость:").pack(side="left", padx=(20, 4))
        ttk.Scale(toolbar, from_=1, to=20, variable=self.speed, length=160,
                  command=self._speed_changed).pack(side="left")
        self.speed_label = ttk.Label(toolbar, text="5 фаз/с", width=14)
        self.speed_label.pack(side="left", padx=6)
        ttk.Checkbutton(toolbar, text="Сетка", variable=self.grid,
                        command=self._draw).pack(side="left", padx=10)
        ttk.Label(toolbar, textvariable=self.generation).pack(side="right")

        settings = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        settings.grid(row=1, column=0, sticky="ew")
        for label, variable, width in (("Ширина", self.width, 5), ("Высота", self.height, 5),
                                       ("Плотность", self.density, 7), ("Seed", self.seed, 12)):
            ttk.Label(settings, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(settings, textvariable=variable, width=width).pack(side="left", padx=(0, 12))
        ttk.Button(settings, text="Создать поле", command=self.new_field).pack(side="left")
        ttk.Button(settings, text="Случайный seed", command=self.random_field).pack(side="left", padx=6)
        ttk.Label(settings, text="Настройки применяются при создании поля.").pack(side="left", padx=10)

        rules_bar = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        rules_bar.grid(row=2, column=0, sticky="ew")
        ttk.Label(rules_bar, text="Шанс случайной точки за ход (0–1):").pack(side="left")
        ttk.Entry(rules_bar, textvariable=self.birth_chance, width=7).pack(side="left", padx=8)
        ttk.Button(rules_bar, text="Применить", command=self.apply_birth_chance).pack(side="left")
        ttk.Label(rules_bar, text="Одна попытка за ход. Действует со следующего шага.").pack(side="left", padx=10)
        ttk.Label(rules_bar, text="Шанс растворения:").pack(side="left", padx=(12, 4))
        ttk.Entry(rules_bar, textvariable=self.dissolve_chance, width=8).pack(side="left")

        agents_bar = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        agents_bar.grid(row=3, column=0, sticky="ew")
        ttk.Label(agents_bar, text="Агенты:").pack(side="left")
        selector = ttk.Combobox(agents_bar, state="readonly", textvariable=self.agent_mode,
                               values=("Выключены", "Случайные", "Эвристика", "Нейросеть", "Нейросеть + поиск"), width=20)
        selector.pack(side="left", padx=6)
        selector.bind("<<ComboboxSelected>>", self._change_agents)
        ttk.Button(agents_bar, text="Загрузить веса…", command=self.load_model).pack(side="left")
        ttk.Label(agents_bar, textvariable=self.agent_status).pack(side="left", padx=10)

        cutoff_bar = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        cutoff_bar.grid(row=4, column=0, sticky="ew")
        ttk.Label(cutoff_bar, text="Выключить автомат через N ходов:").pack(side="left")
        ttk.Scale(cutoff_bar, from_=0, to=200, variable=self.automaton_rounds, length=220,
                  command=self._change_automaton_limit).pack(side="left", padx=8)
        ttk.Label(cutoff_bar, textvariable=self.automaton_limit_text, width=18).pack(side="left")
        ttk.Checkbutton(cutoff_bar, text="Без ограничения", variable=self.automaton_unlimited,
                        command=self._change_automaton_limit).pack(side="left", padx=8)
        ttk.Label(cutoff_bar, text="0 — выключен сразу. Растворение и доход работают всегда.").pack(side="left")

        body = ttk.Frame(self.root, padding=(12, 0))
        body.grid(row=5, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, background=BACKGROUND, highlightthickness=1,
                                highlightbackground="#cbd5e1")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<Motion>", self._inspect)
        self.canvas.bind("<Leave>", lambda _: self.hover.set(""))

        footer = ttk.Frame(self.root, padding=12)
        footer.grid(row=6, column=0, sticky="ew")
        legend = ttk.Frame(footer)
        legend.pack(anchor="w")
        for color, name in ((Color.RED, "Красные"), (Color.BLUE, "Синие")):
            sample = tk.Canvas(legend, width=24, height=20, highlightthickness=0,
                               background=TERRITORY[color])
            sample.create_oval(7, 5, 17, 15, fill=POINTS[color], outline="")
            sample.pack(side="left", padx=(0, 4))
            ttk.Label(legend, text=name).pack(side="left", padx=(0, 16))
        ttk.Label(legend, text="Кружок — точка; заливка — территория; яркий оттенок — захват этого хода.").pack(side="left")
        players = ttk.Frame(footer)
        players.pack(fill="x", pady=(8, 2))
        blue_panel = ttk.LabelFrame(players, text="Синий игрок", padding=8)
        blue_panel.pack(side="left", fill="x", expand=True, padx=(0, 8))
        red_panel = ttk.LabelFrame(players, text="Красный игрок", padding=8)
        red_panel.pack(side="right", fill="x", expand=True)
        ttk.Label(blue_panel, textvariable=self.blue_statistics, foreground=POINTS[Color.BLUE]).pack(anchor="w")
        ttk.Label(red_panel, textvariable=self.red_statistics, foreground=POINTS[Color.RED]).pack(anchor="w")
        for variable in (self.events, self.hover, self.status):
            ttk.Label(footer, textvariable=variable).pack(anchor="w", pady=(4, 0))

    def _reset_agents(self):
        mode = self.agent_mode.get()
        if mode in ("Нейросеть", "Нейросеть + поиск"):
            factory = SearchAgent if mode == "Нейросеть + поиск" else NeuralAgent
            options = dict(config=self.search_config) if factory is SearchAgent else {}
            self.agents = {c: factory(TinyPolicy.from_dict(self.policies[c].as_dict(), int(c))
                                         if self.policies else TinyPolicy(int(c)), seed=int(c), **options)
                           for c in (Color.RED, Color.BLUE)}
            episodes = min(p.episodes for p in self.policies.values()) if self.policies else 0
            legacy = self.policies and any(p.training_version < 3 for p in self.policies.values())
            detail = (f"Поиск: глубина {self.search_config.depth}, луч {self.search_config.beam}. "
                      if factory is SearchAgent else "")
            self.agent_status.set(detail + ("Старые веса: новый обзор и оценка требуют обучения." if legacy else
                                  f"Обучено {episodes} партий; веса заморожены." if episodes else
                                  "Случайные веса; обучения нет."))
        elif mode == "Случайные":
            self.agents = {c: RandomAgent(int(c)) for c in (Color.RED, Color.BLUE)}
            self.agent_status.set("Случайный выбор допустимых действий; обучения нет.")
        else:
            self.agents = {c: HeuristicAgent() for c in (Color.RED, Color.BLUE)}
            self.agent_status.set("Только автомат." if mode == "Выключены" else "Эвристика: правила выбора заданы вручную, обучения нет.")

    def _change_agents(self, _=None):
        if self.phase_busy:
            self.agent_mode.set(self.current_mode)
            messagebox.showinfo("Фаза агента", "Дождитесь завершения текущей фазы.", parent=self.root)
            return
        self.pause()
        self.game.agents_enabled = self.agent_mode.get() != "Выключены"
        if not self.game.agents_enabled:
            while self.game.active_color is not None and not self.game.done:
                self.game.finish_agent_turn()
        self._reset_agents()
        self.refresh()

    def load_model(self):
        if self.phase_busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title="Веса двух агентов", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            policies = load_policies(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            messagebox.showerror("Загрузка весов", str(error), parent=self.root)
            return
        self.policies = policies
        if self.agent_mode.get() != "Нейросеть + поиск":
            self.agent_mode.set("Нейросеть")
        self._change_agents()

    def _speed_changed(self, _=None):
        self.speed_label.configure(text=f"{self.speed.get():.0f} фаз/с")

    def _keyboard_toggle(self, event):
        if event.widget.winfo_class() not in ("TEntry", "Entry", "TScale"):
            self.toggle()
            return "break"

    def _keyboard_step(self, event):
        if event.widget.winfo_class() not in ("TEntry", "Entry", "TScale"):
            self.single_step()
            return "break"

    def toggle(self):
        if self.game.done:
            return
        if self.running:
            self.pause()
        else:
            self.running = True
            self.run_button.configure(text="Пауза")
            self.step_button.configure(state="disabled")
            self.status.set("Автомат работает. Пробел — пауза.")
            if not self.phase_busy:
                self.timer = self.root.after(0, self._tick)

    def pause(self):
        self.running = False
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        self.run_button.configure(text="Запуск")
        self.step_button.configure(state="disabled" if self.phase_busy else "normal")
        self.status.set("Пауза. Пробел — запуск / пауза, → — один шаг.")

    def _advance(self):
        if self.game.simulation is not self.simulation:
            self.game = Game(self.simulation, self.agent_mode.get() != "Выключены", self.round_limit,
                             automaton_rounds=self._automaton_limit())
        if self.game.done:
            self.pause()
            return
        self.phase_statistics_before = self.simulation.board.statistics()
        start = perf_counter()
        if self.game.phase_index != 0:
            self._start_agent_phase()
            return
        phase_result = self.game.advance_automaton()
        self.last_result = phase_result.automaton
        self.step_ms = (perf_counter() - start) * 1000
        self.refresh()

    def _start_agent_phase(self):
        # A UI step still means a whole agent phase. Split internal decisions
        # across Tk callbacks so a large accumulated budget never locks the UI.
        self.phase_busy = True
        self.current_mode = self.agent_mode.get()
        self.step_button.configure(state="disabled")
        self.phase_started = perf_counter()
        self.phase_before = self.simulation.board.copy()
        self.phase_name = self.game.next_phase
        self.phase_color = self.game.active_color
        self.phase_rewards = {Color.RED: 0, Color.BLUE: 0}
        self.phase_moves = 0
        self.phase_timer = self.root.after(1, self._agent_decision)

    def _agent_decision(self):
        self.phase_timer = None
        legal = self.game.legal_actions()
        agent = self.agents[self.phase_color]
        action = agent.choose(self.game, legal)
        if isinstance(agent, SearchAgent):
            self.agent_status.set(f"Поиск: глубина {agent.config.depth}; проверено {agent.last_nodes} переходов; "
                                  f"цепочка {len(agent.last_sequence)}; веса заморожены.")
        reward = self.game.apply_action(action)
        for color in self.phase_rewards:
            self.phase_rewards[color] += reward[color]
        if action.kind != ActionKind.PASS:
            self.phase_moves += 1
            self._show_agent_changes(action)
            self.phase_timer = self.root.after(1, self._agent_decision)
            return
        terminal = self.game.finish_agent_turn()
        for color in self.phase_rewards:
            self.phase_rewards[color] += terminal[color]
        self.phase_busy = False
        self.step_ms = (perf_counter() - self.phase_started) * 1000
        self._show_agent_changes(action)
        if not self.running:
            self.step_button.configure(state="normal")
        self.refresh()
        if self.running and not self.game.done:
            self.timer = self.root.after(max(1, round(1000 / self.speed.get())), self._tick)

    def _show_agent_changes(self, action):
        before, after = self.phase_before, self.simulation.board
        self.game.last_result = PhaseResult(self.phase_name, dict(self.phase_rewards), self.phase_moves, action,
            highlighted_cells=frozenset((r, c) for r, c in after.cells()
                if after.points[r][c] == Color.EMPTY and after.owners[r][c] != Color.EMPTY
                and (before.owners[r][c] != after.owners[r][c] or before.points[r][c] != Color.EMPTY)),
            highlighted_points=frozenset((r, c) for r, c in after.cells()
                if after.points[r][c] != Color.EMPTY and before.points[r][c] != after.points[r][c]))
        self.last_result = self.game.last_result
        self.refresh()

    def _tick(self):
        self.timer = None
        if not self.running or self.phase_busy:
            return
        start = perf_counter()
        self._advance()
        if self.phase_busy or self.game.done:
            return
        elapsed_ms = (perf_counter() - start) * 1000
        # No catch-up loop: allow UI events between steps even on large fields.
        delay = max(1, round(1000 / self.speed.get() - elapsed_ms))
        self.timer = self.root.after(delay, self._tick)

    def single_step(self):
        if not self.running and not self.phase_busy:
            self._advance()

    def restart(self):
        self.pause()
        self._cancel_phase()
        self.simulation = Simulation(self.initial_board, self.simulation.rules)
        self.simulation.random.setstate(self.initial_random_state)
        self.game = Game(self.simulation, self.agent_mode.get() != "Выключены", self.round_limit,
                         automaton_rounds=self._automaton_limit())
        self._reset_agents()
        self.last_result = None
        self.phase_statistics_before = self.simulation.board.statistics()
        self.step_ms = 0
        self.refresh()

    def _automaton_limit(self):
        return None if self.automaton_unlimited.get() else round(self.automaton_rounds.get())

    def _change_automaton_limit(self, _=None):
        limit = self._automaton_limit()
        self.game.automaton_rounds = limit
        self.automaton_limit_text.set("Без ограничения" if limit is None else f"{limit} раундов")
        if hasattr(self, "canvas"):
            self.refresh()

    def _read_birth_chance(self):
        chance = float(self.birth_chance.get().replace(",", "."))
        if not 0 <= chance <= 1:
            raise ValueError("Шанс рождения должен быть от 0 до 1.")
        return chance

    def apply_birth_chance(self):
        try:
            chance = self._read_birth_chance()
            dissolve = float(self.dissolve_chance.get().replace(",", "."))
            if not 0 <= dissolve <= 1:
                raise ValueError("Шанс растворения должен быть от 0 до 1.")
        except ValueError as error:
            messagebox.showerror("Шанс рождения", str(error), parent=self.root)
            return
        self.simulation.rules = replace(self.simulation.rules, random_birth_chance=chance, interior_dissolve_chance=dissolve)
        self.status.set(f"Шанс случайной точки: {chance:g}. Применён со следующего хода.")

    def new_field(self):
        try:
            width, height = int(self.width.get()), int(self.height.get())
            density = float(self.density.get().replace(",", "."))
            seed = int(self.seed.get())
            chance = self._read_birth_chance()
            dissolve = float(self.dissolve_chance.get().replace(",", "."))
            if not 0 <= dissolve <= 1:
                raise ValueError("Шанс растворения должен быть от 0 до 1.")
            if not (1 <= width <= 200 and 1 <= height <= 200):
                raise ValueError("Ширина и высота должны быть от 1 до 200.")
            if not 0 <= density <= 1:
                raise ValueError("Плотность должна быть от 0 до 1.")
        except ValueError as error:
            messagebox.showerror("Параметры поля", f"Проверьте введённые значения.\n{error}", parent=self.root)
            return
        self.pause()
        self._cancel_phase()
        self.simulation = Simulation.random_field(width, height, density,
                                                  rules=replace(self.simulation.rules, random_birth_chance=chance,
                                                                interior_dissolve_chance=dissolve), seed=seed)
        self.game = Game(self.simulation, self.agent_mode.get() != "Выключены", self.round_limit,
                         automaton_rounds=self._automaton_limit())
        self._reset_agents()
        self.initial_board = self.simulation.board.copy()
        self.initial_random_state = self.simulation.random.getstate()
        self.last_result = None
        self.phase_statistics_before = self.simulation.board.statistics()
        self.step_ms = 0
        self.geometry = None
        self.hover.set("Наведите курсор на клетку, чтобы увидеть её состояние.")
        self.refresh()

    def random_field(self):
        self.seed.set(str(SystemRandom().randrange(2**32)))
        self.new_field()

    def refresh(self):
        state = "автомат включён" if self.game.automaton_active else "только растворение и доход"
        self.generation.set(f"Раунд {self.game.round}/{self.round_limit} · {state} · Далее: {self.game.next_phase}")
        stats = self.simulation.board.statistics()
        self.statistics.set("    |    ".join(
            f"{name}: точек {stats[key]['points']}, территория {stats[key]['territory']}, "
            f"без точек {stats[key]['empty_territory']}"
            for key, name in (("red", "Красные"), ("blue", "Синие"))))
        for color, variable in ((Color.BLUE, self.blue_statistics), (Color.RED, self.red_statistics)):
            p, s = self.game.players[color], stats[color.name.lower()]
            before = self.phase_statistics_before[color.name.lower()]
            def metric(key):
                return f"{s[key]} ({s[key] - before[key]:+d})"
            variable.set(f"Очки: {p.score:+d}    Действий: {p.budget}    Доход: +{p.income}/раунд\n"
                         f"Потрачено в раунде: {p.spent}\n"
                         f"Точки: {metric('points')}    Территория: {metric('territory')}\n"
                         f"Без точек: {metric('empty_territory')}")
        if self.last_result is None:
            self.events.set("Начальное состояние. Автомат остановлен.")
        elif isinstance(self.last_result, PhaseResult):
            e = self.last_result
            self.events.set(f"{e.phase}: выполнено действий {e.action_count}; "
                            f"награда синих {e.rewards[Color.BLUE]:+d}, красных {e.rewards[Color.RED]:+d}.")
        else:
            e = self.last_result
            self.events.set(f"За такт: родилось {e.births}, погибло {e.deaths}, "
                            f"перекрашено {e.conversions}; захвачено клеток {e.captured_cells}, "
                            f"точек {e.captured_points}; растворено {e.dissolved_points}. Расчёт: {self.step_ms:.1f} мс.")
        if self.game.done:
            self.pause()
            name = "синие" if self.game.winner == Color.BLUE else "красные" if self.game.winner == Color.RED else None
            self.status.set(f"Партия завершена. Победили {name}." if name else "Партия завершена. Ничья.")
        self._draw()

    def _resize(self, _):
        if self.render_timer is not None:
            self.root.after_cancel(self.render_timer)
        self.render_timer = self.root.after(40, self._redraw)

    def _redraw(self):
        self.render_timer = None
        self._draw()

    def _draw(self):
        board = self.simulation.board
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 10 or height < 10:
            return
        size = min((width - 16) / board.width, (height - 16) / board.height)
        x0, y0 = (width - size * board.width) / 2, (height - size * board.height) / 2
        geometry = (board.width, board.height, width, height)
        if geometry != self.geometry:
            self.canvas.delete("all")
            self.rectangles, self.dots = [], []
            for r, c in board.cells():
                x, y = x0 + c * size, y0 + r * size
                self.rectangles.append(self.canvas.create_rectangle(
                    x, y, x + size, y + size, width=0))
                margin = size * 0.24
                self.dots.append(self.canvas.create_oval(
                    x + margin, y + margin, x + size - margin, y + size - margin,
                    outline="", state="hidden"))
            self.canvas.create_rectangle(x0, y0, x0 + board.width * size,
                                         y0 + board.height * size, outline="#94a3b8", tags="border")
            self.geometry = geometry
        for index, (r, c) in enumerate(board.cells()):
            owner = board.owners[r][c]
            is_new_cell = self.last_result is not None and (r, c) in self.last_result.highlighted_cells
            fill = NEW_TERRITORY[owner] if is_new_cell else TERRITORY[owner]
            self.canvas.itemconfigure(self.rectangles[index], fill=fill)
            point = board.points[r][c]
            if point:
                is_new_point = self.last_result is not None and (r, c) in self.last_result.highlighted_points
                fill = NEW_POINTS[point] if is_new_point else POINTS[point]
                self.canvas.itemconfigure(self.dots[index], fill=fill, state="normal")
            else:
                self.canvas.itemconfigure(self.dots[index], state="hidden")
        self.canvas.delete("grid")
        if self.grid.get() and size >= 5:
            for col in range(1, board.width):
                x = x0 + col * size
                self.canvas.create_line(x, y0, x, y0 + board.height * size,
                                        fill="#cbd5e1", tags="grid")
            for row in range(1, board.height):
                y = y0 + row * size
                self.canvas.create_line(x0, y, x0 + board.width * size, y,
                                        fill="#cbd5e1", tags="grid")
        self.canvas.tag_raise("border")
        self.cell_geometry = (x0, y0, size)

    def _inspect(self, event):
        if not hasattr(self, "cell_geometry"):
            return
        x0, y0, size = self.cell_geometry
        r, c = int((event.y - y0) // size), int((event.x - x0) // size)
        board = self.simulation.board
        if not (0 <= r < board.height and 0 <= c < board.width):
            self.hover.set("")
            return
        names = {Color.EMPTY: "нет", Color.RED: "красная", Color.BLUE: "синяя"}
        self.hover.set(f"Строка {r}, столбец {c} · точка: {names[board.points[r][c]]} · "
                       f"территория: {names[board.owners[r][c]]}")

    def close(self):
        self.pause()
        self._cancel_phase()
        if self.render_timer is not None:
            self.root.after_cancel(self.render_timer)
        self.root.destroy()

    def _cancel_phase(self):
        if self.phase_timer is not None:
            self.root.after_cancel(self.phase_timer)
            self.phase_timer = None
        self.phase_busy = False
        self.step_button.configure(state="normal")


def launch(simulation: Simulation, density: float, seed: int, round_limit=200, mode="Эвристика", model=None,
           search_config=None, automaton_rounds=None):
    root = tk.Tk()
    app = SimulationApp(root, simulation)
    app.automaton_unlimited.set(automaton_rounds is None)
    app.automaton_rounds.set(8 if automaton_rounds is None else automaton_rounds)
    app._change_automaton_limit()
    app.search_config = search_config or SearchConfig()
    app.density.set(str(density))
    app.seed.set(str(seed))
    app.round_limit = round_limit
    app.game.round_limit = round_limit
    if model:
        app.policies = load_policies(model)
        if mode != "Нейросеть + поиск":
            mode = "Нейросеть"
    app.agent_mode.set(mode)
    app._change_agents()
    root.mainloop()
