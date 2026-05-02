# ============================================================
#  env.py  —  DroneAgentEnv  (one per drone)
#
#  Each drone gets its own Gymnasium-compatible environment
#  that shares a single SharedSimulation instance.
#  All 10 envs are synchronised — physics steps only after
#  every active drone has submitted its action.
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    OBS_SIZE, NUM_NEIGHBOURS_OBS,
    WIDTH, HEIGHT, MAX_SPEED, FOV_RANGE,
)
from simulation import SharedSimulation, S_FAILED, S_ARRIVED
from reward import compute_reward


class DroneAgentEnv(gym.Env):
    """
    Single-drone Gymnasium environment backed by SharedSimulation.

    Observation vector (20 values)
    ───────────────────────────────
    [0-1]   own velocity            (normalised by MAX_SPEED)
    [2-3]   own position            (normalised by W, H)
    [4-5]   relative target pos     (normalised by W, H)
    [6]     distance to target      (normalised by diagonal)
    [7-18]  3 nearest neighbours:   rel_x, rel_y, rel_vx, rel_vy
            (zero-padded when fewer than NUM_NEIGHBOURS_OBS visible)
    [19]    min distance to any neighbour (normalised by FOV_RANGE)

    Action space (continuous)
    ─────────────────────────
    [ax, ay] ∈ [-1, 1]  — normalised thrust vector
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, drone_idx: int, sim: SharedSimulation):
        super().__init__()
        self.drone_idx = drone_idx
        self.sim       = sim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self._max_dist = np.sqrt(WIDTH**2 + HEIGHT**2)

    # ── Gymnasium API ────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Env 0 is responsible for resetting the shared sim
        if self.drone_idx == 0:
            target = options.get("target_pos") if options else None
            self.sim.reset(target_pos=target)
        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        d = self.sim.get_drone(self.drone_idx)

        # if this drone is already done, skip and return terminal obs
        if d.status in (S_FAILED, S_ARRIVED):
            obs     = self._get_obs()
            reward  = 0.0
            terminated = True
            truncated  = False
            return obs, reward, terminated, truncated, {}

        # push action into shared sim (may trigger physics step)
        self.sim.push_action(self.drone_idx, action)

        obs        = self._get_obs()
        reward     = compute_reward(self.drone_idx, self.sim)
        terminated = (d.status in (S_FAILED, S_ARRIVED)
                      or self.sim.episode_done)
        truncated  = False
        info       = {
            "arrived":  d.status == S_ARRIVED,
            "failed":   d.status == S_FAILED,
            "all_arrived": self.sim.all_arrived,
        }
        return obs, reward, terminated, truncated, info

    # ── Observation builder ──────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        d   = self.sim.get_drone(self.drone_idx)
        obs = []

        # own state
        obs += [
            d.velocity[0] / MAX_SPEED,
            d.velocity[1] / MAX_SPEED,
            (d.position[0] / WIDTH)  * 2 - 1,
            (d.position[1] / HEIGHT) * 2 - 1,
        ]

        # target info
        rel_target = sim_target = self.sim.target - d.position
        obs += [
            np.clip(rel_target[0] / WIDTH,  -1, 1),
            np.clip(rel_target[1] / HEIGHT, -1, 1),
            np.clip(np.linalg.norm(rel_target) / self._max_dist, 0, 1),
        ]

        # nearest neighbours (within FOV)
        neighbours = self._get_neighbours(d)
        min_dist   = 1.0
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

        # minimum neighbour distance
        obs += [min_dist]

        return np.array(obs, dtype=np.float32)

    def _get_neighbours(self, d) -> list:
        """Return up to NUM_NEIGHBOURS_OBS nearest alive drones, sorted by distance."""
        others = [
            x for x in self.sim.drones
            if x.idx != d.idx and x.alive
        ]
        others.sort(key=lambda x: np.linalg.norm(d.position - x.position))
        return others[:NUM_NEIGHBOURS_OBS]
