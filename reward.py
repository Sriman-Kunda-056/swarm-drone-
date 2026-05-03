# ============================================================
#  reward.py  —  5-component reward function
# ============================================================

import numpy as np
from config import (
    WIDTH, HEIGHT,
    TARGET_RADIUS,
    R_TARGET_PROGRESS, R_SLOT_PROGRESS,
    R_TARGET_CLOSE, R_SLOT_CLOSE, R_FORMATION,
    R_SEPARATION, R_ALIGNMENT,
    R_COLLISION, R_ALIVE, R_ARRIVAL_BONUS,
    R_SLOT_REACHED, R_TARGET_REACHED,
)
from simulation import S_FAILED

_MAX_DIST = np.sqrt(WIDTH**2 + HEIGHT**2)


def compute_reward(drone_idx: int, sim,
                   prev_target_dist: float | None = None,
                   prev_slot_dist: float | None = None) -> float:
    d = sim.get_drone(drone_idx)
    if d.status == S_FAILED:
        return 0.0

    reward = 0.0

    # 1. Target goal: reward movement toward the shared destination
    dist_target = np.linalg.norm(d.position - sim.target)
    norm_target = np.clip(dist_target / _MAX_DIST, 0.0, 1.0)
    if prev_target_dist is not None:
        reward += R_TARGET_PROGRESS * ((prev_target_dist - dist_target) / _MAX_DIST)
    reward += R_TARGET_CLOSE * (1.0 - norm_target)
    if dist_target <= TARGET_RADIUS:
        reward += R_TARGET_REACHED

    # 2. Formation goal: each drone has an assigned slot around the target
    slot = sim.get_formation_slot(drone_idx)
    dist_slot = np.linalg.norm(d.position - slot)
    norm_slot = np.clip(dist_slot / _MAX_DIST, 0.0, 1.0)
    if prev_slot_dist is not None:
        reward += R_SLOT_PROGRESS * ((prev_slot_dist - dist_slot) / _MAX_DIST)
    reward += R_SLOT_CLOSE * (1.0 - norm_slot)

    # 3. Cohesion: keep the swarm compact so the formation stays coupled
    alive_pos = np.array([x.position for x in sim.drones if x.alive])
    if len(alive_pos) > 1:
        centroid  = alive_pos.mean(axis=0)
        mean_spread = np.mean(np.linalg.norm(alive_pos - centroid, axis=1))
        cohesion = 1.0 - np.clip(mean_spread / (_MAX_DIST * 0.35), 0.0, 1.0)
        reward += R_FORMATION * cohesion

    # 4. Boids-style separation and alignment
    neighbours = [x for x in sim.drones if x.idx != drone_idx and x.alive]
    if neighbours:
        nearest = min(np.linalg.norm(d.position - n.position) for n in neighbours)
        safe_dist = 32.0
        separation = np.clip((nearest - safe_dist) / safe_dist, -1.0, 1.0)
        # Positive when drones are comfortably spaced, negative when too close.
        reward += R_SEPARATION * separation

        nearest_n = min(neighbours, key=lambda n: np.linalg.norm(d.position - n.position))
        speed_d = np.linalg.norm(d.velocity)
        speed_n = np.linalg.norm(nearest_n.velocity)
        if speed_d > 1e-5 and speed_n > 1e-5:
            alignment = float(np.dot(d.velocity / speed_d, nearest_n.velocity / speed_n))
            reward += R_ALIGNMENT * alignment

    # 5. Collision penalty
    if sim.collisions.get(drone_idx, False):
        reward += R_COLLISION

    # 6. Alive bonus
    reward += R_ALIVE

    # 7. Slot reached bonus
    if d.at_slot and not getattr(d, '_slot_rewarded', False):
        reward += R_SLOT_REACHED
        d._slot_rewarded = True

    # 8. Full swarm at slots
    if sim.all_at_slots:
        reward += R_ARRIVAL_BONUS

    return float(reward)
