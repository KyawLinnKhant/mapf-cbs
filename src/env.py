"""Multi-agent MAPF environment with lifelong goals and CBS oracle shaping."""

from __future__ import annotations

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cbs import cbs
from .grid import Grid, Position, MOVES
from .maps import MapConfig, generate_map, sample_positions
from .dynamic import DynamicObstacle, spawn_dynamic_obstacles

# ── Observation constants ────────────────────────────────────────────────────
CROP_SIZE = 7            # local grid crop (square)
CROP_CHANNELS = 3        # obstacles | other_agents | goals
GOAL_DIM = 2             # normalised (dx, dy) to own goal
TIME_DIM = 1             # normalised timestep t / max_steps
OBS_DIM = CROP_SIZE * CROP_SIZE * CROP_CHANNELS + GOAL_DIM + TIME_DIM   # 150

N_ACTIONS = 5            # wait, right, left, up, down  (matches MOVES order)


class MAPFEnv:
    """
    Decentralised multi-agent MAPF environment.

    API
    ---
    reset()  → dict[agent_id, obs_array]
    step(actions: dict[agent_id, int])
             → obs, rewards, dones, info

    Observation (per agent, flat float32 vector of length OBS_DIM=150)
    -------------------------------------------------------------------
    [0:147]   3-channel 7×7 local crop centred on agent
              ch0 = obstacles,  ch1 = other agents,  ch2 = all goals
    [147:149] normalised (dx, dy) from agent to own goal
    [149]     normalised timestep

    Lifelong goals
    --------------
    When an agent reaches its goal it immediately receives a new random goal.
    """

    def __init__(
        self,
        config: MapConfig,
        max_steps: int = 256,
        cbs_timeout: float = 1.5,
        seed: Optional[int] = None,
        n_dynamic_obstacles: int = 0,
        dynamic_pattern: str = "mixed",
    ):
        self.config = config
        self.max_steps = max_steps
        self.cbs_timeout = cbs_timeout
        self._rng = np.random.default_rng(seed)
        self.n_dynamic_obstacles = n_dynamic_obstacles
        self.dynamic_pattern = dynamic_pattern

        self.n_agents = config.n_agents
        self.grid: Optional[Grid] = None
        self.grid_array: Optional[np.ndarray] = None

        # Per-episode state
        self.positions: Dict[int, Position] = {}
        self.goals: Dict[int, Position] = {}
        self.t: int = 0
        self.goals_reached: Dict[int, int] = {}
        self.collisions: int = 0
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # CBS oracle (best-effort)
        self.cbs_paths: Optional[Dict[int, List[Position]]] = None
        self.cbs_available: bool = False

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, _retry: int = 0) -> Dict[int, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if _retry > 10:
            raise RuntimeError("MAPFEnv.reset: could not generate a valid map after 10 retries")

        map_seed = int(self._rng.integers(0, 2 ** 31))
        self.grid, self.grid_array = generate_map(self.config, seed=map_seed)
        # Cache padded obstacle array for fast crop extraction
        self._padded_grid = np.pad(self.grid_array, CROP_SIZE // 2,
                                   mode='constant', constant_values=1)

        starts = sample_positions(self.grid, self.n_agents, self._rng)
        if starts is None:
            return self.reset(_retry=_retry + 1)

        goals = sample_positions(self.grid, self.n_agents, self._rng, exclude=starts)
        if goals is None:
            return self.reset(_retry=_retry + 1)

        self.positions = {i: starts[i] for i in range(self.n_agents)}
        self.goals = {i: goals[i] for i in range(self.n_agents)}
        self.t = 0
        self.goals_reached = {i: 0 for i in range(self.n_agents)}
        self.collisions = 0

        # Spawn dynamic obstacles on free cells not used by agents
        occupied = list(starts) + list(goals)
        self.dynamic_obstacles = spawn_dynamic_obstacles(
            self.n_dynamic_obstacles,
            self.grid,
            self._rng,
            exclude=occupied,
            pattern=self.dynamic_pattern,
        )

        self._run_cbs_oracle(starts, goals)

        return self._all_obs()

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self, actions: Dict[int, int]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], dict]:

        prev_pos = dict(self.positions)
        prev_dist = {i: self._manhattan(self.positions[i], self.goals[i])
                     for i in range(self.n_agents)}

        # 0. Move dynamic obstacles (they yield to current agent positions)
        agent_cells = set(self.positions.values())
        for obs in self.dynamic_obstacles:
            obs.step(occupied=agent_cells)
        dyn_cells = {obs.pos for obs in self.dynamic_obstacles}

        # 1. Intended positions (wall check + dynamic obstacle check)
        intended: Dict[int, Position] = {}
        for i, a in actions.items():
            dx, dy = MOVES[a]
            x, y = self.positions[i]
            nx, ny = x + dx, y + dy
            if self.grid.is_free(nx, ny) and (nx, ny) not in dyn_cells:
                intended[i] = (nx, ny)
            else:
                intended[i] = (x, y)
                if (nx, ny) in dyn_cells:
                    self.collisions += 1  # collision with dynamic obstacle

        # 2. Vertex conflict: multiple agents targeting same cell → all stay
        cell_count = Counter(intended.values())
        for i in range(self.n_agents):
            if cell_count[intended[i]] > 1:
                intended[i] = prev_pos[i]
                self.collisions += 1

        # 3. Edge conflict (swap): agents crossing paths → both stay
        for a in range(self.n_agents):
            for b in range(a + 1, self.n_agents):
                if intended[a] == prev_pos[b] and intended[b] == prev_pos[a]:
                    intended[a] = prev_pos[a]
                    intended[b] = prev_pos[b]
                    self.collisions += 1

        # 4. Commit moves
        for i in range(self.n_agents):
            self.positions[i] = intended[i]

        self.t += 1

        # 5. Rewards
        rewards: Dict[int, float] = {}
        for i in range(self.n_agents):
            r = -0.01  # per-step penalty

            # Blocked-by-collision penalty (tried to move, stayed)
            if self.positions[i] == prev_pos[i] and actions[i] != 0:
                r -= 0.1

            # Distance shaping
            curr_dist = self._manhattan(self.positions[i], self.goals[i])
            r += 0.05 * (prev_dist[i] - curr_dist)

            # Goal reached
            if self.positions[i] == self.goals[i]:
                r += 1.0
                self.goals_reached[i] += 1
                # Lifelong: assign new random goal immediately
                new_goal = sample_positions(
                    self.grid, 1, self._rng,
                    exclude=list(self.positions.values()),
                )
                if new_goal:
                    self.goals[i] = new_goal[0]
                    # CBS oracle paths are now stale; disable shaping
                    self.cbs_available = False

            rewards[i] = r

        done = self.t >= self.max_steps
        dones = {i: done for i in range(self.n_agents)}
        info = {
            "goals_reached": sum(self.goals_reached.values()),
            "collisions": self.collisions,
            "makespan": self.t,
            "success": sum(v > 0 for v in self.goals_reached.values()),
        }

        return self._all_obs(), rewards, dones, info

    # ── CBS oracle ────────────────────────────────────────────────────────────

    # Hard budget for CBS oracle calls during training.
    # max_t=64, CT nodes=200, A* nodes=2000 → completes in <2ms or bails out fast.
    _CBS_ORACLE_MAX_T        = 64
    _CBS_ORACLE_MAX_CT       = 100
    _CBS_ORACLE_MAX_ASTAR    = 500

    def _run_cbs_oracle(self, starts: List[Position], goals: List[Position]) -> None:
        """
        Run CBS synchronously with hard node-expansion caps.
        Guaranteed to return in bounded time regardless of map topology.
        """
        try:
            solution = cbs(
                self.grid, starts, goals,
                max_t          = self._CBS_ORACLE_MAX_T,
                max_ct_nodes   = self._CBS_ORACLE_MAX_CT,
                max_astar_nodes= self._CBS_ORACLE_MAX_ASTAR,
            )
            if solution is not None:
                self.cbs_paths = solution
                self.cbs_available = True
                return
        except Exception:
            pass
        self.cbs_paths = None
        self.cbs_available = False

    def get_cbs_action(self, agent_id: int) -> Optional[int]:
        """CBS-suggested action at current timestep, or None."""
        if not self.cbs_available or self.cbs_paths is None:
            return None
        path = self.cbs_paths.get(agent_id)
        if path is None or self.t >= len(path) - 1:
            return None
        curr = path[self.t]
        nxt = path[self.t + 1]
        move = (nxt[0] - curr[0], nxt[1] - curr[1])
        for action, m in enumerate(MOVES):
            if m == move:
                return action
        return 0

    # ── Observations ──────────────────────────────────────────────────────────

    def _all_obs(self) -> Dict[int, np.ndarray]:
        return {i: self._obs(i) for i in range(self.n_agents)}

    def _obs(self, agent_id: int) -> np.ndarray:
        crop = self._crop(agent_id).flatten()          # 147
        gvec = self._goal_vec(agent_id)                # 2
        t_norm = np.array([self.t / self.max_steps], dtype=np.float32)  # 1
        return np.concatenate([crop, gvec, t_norm])    # 150

    def _crop(self, agent_id: int) -> np.ndarray:
        """3×7×7 local crop centred on the agent — vectorised with numpy slicing."""
        H, W = self.grid.height, self.grid.width
        half = CROP_SIZE // 2
        cx, cy = self.positions[agent_id]

        # World coordinates of the crop window
        x0, x1 = cx - half, cx + half + 1   # cols
        y0, y1 = cy - half, cy + half + 1   # rows

        # Obstacle channel — use cached padded grid (cx,cy → cx, cy in padded coords)
        py0, py1 = cy, cy + CROP_SIZE
        px0, px1 = cx, cx + CROP_SIZE
        obs_ch = self._padded_grid[py0:py1, px0:px1].astype(np.float32)

        # Agent channel (other MARL agents = 1.0, dynamic obstacles = 0.5)
        agent_ch = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        for other_id, (ox, oy) in self.positions.items():
            if other_id == agent_id:
                continue
            col = ox - x0
            row = oy - y0
            if 0 <= row < CROP_SIZE and 0 <= col < CROP_SIZE:
                agent_ch[row, col] = 1.0
        for dyn_obs in self.dynamic_obstacles:
            ox, oy = dyn_obs.pos
            col = ox - x0
            row = oy - y0
            if 0 <= row < CROP_SIZE and 0 <= col < CROP_SIZE:
                agent_ch[row, col] = 0.5  # distinct from MARL agents

        # Goal channel
        goal_ch = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        for (gx, gy) in self.goals.values():
            col = gx - x0
            row = gy - y0
            if 0 <= row < CROP_SIZE and 0 <= col < CROP_SIZE:
                goal_ch[row, col] = 1.0

        return np.stack([obs_ch, agent_ch, goal_ch])  # [3, 7, 7]

    def _goal_vec(self, agent_id: int) -> np.ndarray:
        ax, ay = self.positions[agent_id]
        gx, gy = self.goals[agent_id]
        scale = max(self.grid.width, self.grid.height)
        return np.array([(gx - ax) / scale, (gy - ay) / scale], dtype=np.float32)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS
