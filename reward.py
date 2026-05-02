# ============================================================
#  reward.py  —  Reward function (4 components, all tunable)
#
#  Called once per step per drone by DroneAgentEnv.
#  All weights live in config.py.
# ============================================================

import numpy as np
from config import (
    WIDTH, HEIGHT,
    R_APPROACH, R_FORMATION, R_COLLISION,
    R_ALIVE, R_ARRIVAL_BONUS, R_PARTIAL_ARRIVAL,
    TARGET_RADIUS,
)
from simulation import S_ARRIVED, S_FAILED


_MAX_DIST = np.sqrt(WIDTH**2 + HEIGHT**2)


def compute_reward(drone_idx: int, sim) -> float:
    """
    Compute total reward for one drone for the current step.

    Components
    ──────────
    1. Approach     — proportional to how close this drone is to target
    2. Formation    — how tightly the active swarm clusters together
    3. Collision    — hard penalty when inside danger radius of another
    4. Alive bonus  — small positive per surviving step
    5. Partial arr  — one-off when THIS drone enters target zone
    6. Arrival bon  — large one-off when ENTIRE remaining swarm arrives
    """
    d = sim.get_drone(drone_idx)

    # dead drones get no reward
    if d.status == S_FAILED:
        return 0.0

    reward = 0.0

    # ── 1. Approach reward ───────────────────────────────────
    dist_to_target = np.linalg.norm(d.position - sim.target)
    reward += R_APPROACH * (dist_to_target / _MAX_DIST)

    # ── 2. Formation reward ──────────────────────────────────
    active_pos = np.array([
        x.position for x in sim.drones
        if x.alive and x.status != S_ARRIVED
    ])
    if len(active_pos) > 1:
        centroid   = active_pos.mean(axis=0)
        dists      = np.linalg.norm(active_pos - centroid, axis=1)
        formation  = np.std(dists) / _MAX_DIST
        reward += R_FORMATION * formation

    # ── 3. Collision penalty ─────────────────────────────────
    if sim.collisions.get(drone_idx, False):
        reward += R_COLLISION

    # ── 4. Alive bonus ───────────────────────────────────────
    reward += R_ALIVE

    # ── 5. Partial arrival (this drone just arrived) ─────────
    if drone_idx in sim.new_arrivals:
        reward += R_PARTIAL_ARRIVAL

    # ── 6. Full swarm arrival bonus ──────────────────────────
    if sim.all_arrived:
        reward += R_ARRIVAL_BONUS

    return float(reward)
