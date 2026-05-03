# ============================================================
#  env.py  —  DroneAgentEnv  (one per drone, obs size = 22)
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    OBS_SIZE, NUM_NEIGHBOURS_OBS,
    WIDTH, HEIGHT, MAX_SPEED, FOV_RANGE,
)
from simulation import SharedSimulation, S_FAILED


class DroneAgentEnv(gym.Env):
    """
    Observation vector (22 values)
    [0-1]   own velocity           (÷ MAX_SPEED)
    [2-3]   own position           (normalised to [-1,1])
    [4-5]   relative target pos    (÷ W, H)
    [6]     distance to target     (÷ diagonal)
    [7-18]  3 nearest neighbours   rel_x, rel_y, rel_vx, rel_vy each
    [19]    min distance to neighbour
    [20-21] relative slot position (÷ W, H)  ← NEW
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, drone_idx: int, sim: SharedSimulation):
        super().__init__()
        self.drone_idx = drone_idx
        self.sim       = sim
        self._max_dist = np.sqrt(WIDTH**2 + HEIGHT**2)
        self._prev_target_dist: float | None = None
        self._prev_slot_dist: float | None = None

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.drone_idx == 0:
            target = options.get("target_pos") if options else None
            self.sim.reset(target_pos=target)
        self._prev_target_dist = None
        self._prev_slot_dist = None
        self._sync_reward_baseline()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        d = self.sim.get_drone(self.drone_idx)
        if d.status == S_FAILED:
            return self._get_obs(), 0.0, True, False, {}

        self.sim.push_action(self.drone_idx, action)

        from reward import compute_reward
        obs        = self._get_obs()
        reward     = compute_reward(
            self.drone_idx,
            self.sim,
            self._prev_target_dist,
            self._prev_slot_dist,
        )
        self._sync_reward_baseline()
        terminated = d.status == S_FAILED or self.sim.episode_done
        return obs, reward, terminated, False, {
            "at_slot":     d.at_slot,
            "all_at_slots": self.sim.all_at_slots,
            "goal":         self.sim.get_drone_goal(self.drone_idx),
        }

    def _sync_reward_baseline(self) -> None:
        d = self.sim.get_drone(self.drone_idx)
        self._prev_target_dist = float(np.linalg.norm(d.position - self.sim.target))
        self._prev_slot_dist = float(np.linalg.norm(d.position - self.sim.get_drone_goal(self.drone_idx)))

    def _get_obs(self) -> np.ndarray:
        d   = self.sim.get_drone(self.drone_idx)
        obs = []

        # own state
        obs += [
            d.velocity[0] / MAX_SPEED,
            d.velocity[1] / MAX_SPEED,
            d.position[0] / WIDTH  * 2 - 1,
            d.position[1] / HEIGHT * 2 - 1,
        ]

        # target
        rel_t = self.sim.target - d.position
        obs  += [
            np.clip(rel_t[0] / WIDTH,  -1, 1),
            np.clip(rel_t[1] / HEIGHT, -1, 1),
            np.clip(np.linalg.norm(rel_t) / self._max_dist, 0, 1),
        ]

        # neighbours
        neighbours = sorted(
            [x for x in self.sim.drones if x.idx != self.drone_idx and x.alive],
            key=lambda x: np.linalg.norm(d.position - x.position)
        )[:NUM_NEIGHBOURS_OBS]

        min_dist = 1.0
        for i in range(NUM_NEIGHBOURS_OBS):
            if i < len(neighbours):
                n       = neighbours[i]
                rel_pos = n.position - d.position
                dist_n  = np.linalg.norm(rel_pos)
                min_dist = min(min_dist, dist_n / FOV_RANGE)
                obs += [
                    np.clip(rel_pos[0] / FOV_RANGE, -1, 1),
                    np.clip(rel_pos[1] / FOV_RANGE, -1, 1),
                    n.velocity[0] / MAX_SPEED,
                    n.velocity[1] / MAX_SPEED,
                ]
            else:
                obs += [0.0, 0.0, 0.0, 0.0]

        obs += [min_dist]

        # formation slot (NEW)
        slot     = self.sim.get_formation_slot(self.drone_idx)
        rel_slot = slot - d.position
        obs += [
            np.clip(rel_slot[0] / WIDTH,  -1, 1),
            np.clip(rel_slot[1] / HEIGHT, -1, 1),
        ]

        return np.array(obs, dtype=np.float32)
