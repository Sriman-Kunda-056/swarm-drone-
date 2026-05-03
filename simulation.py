# ============================================================
#  simulation.py  —  Physics engine + formation manager
#
#  Key changes from v1:
#  • Drones NEVER stop — they hover at their formation slot
#  • get_formation_slot() computes per-drone target position
#    for both CIRCLE and LEADER modes
#  • S_ARRIVED is informational only (not a freeze state)
#  • Comm broadcast stored so hud.py can draw links
# ============================================================

import numpy as np
import math
import random
from config import (
    WIDTH, HEIGHT, NUM_DRONES, MAX_SPEED,
    DANGER_RADIUS, SPAWN_RADIUS,
    TARGET_POS, TARGET_RADIUS,
    FAILURE_PROB_PER_STEP, FAILURE_FLICKER_FRAMES,
    MIN_DRONES_TO_CONTINUE, ACTION_SCALE, MAX_STEPS_PER_EP,
    FORMATION_CIRCLE_RADIUS, FORMATION_V_SPACING,
    COMM_RANGE,
)

# ── Drone status ─────────────────────────────────────────────
S_ACTIVE  = "ACTIVE"
S_FAILING = "FAILING"
S_FAILED  = "FAILED"
S_AT_SLOT = "AT_SLOT"    # reached formation slot — still hovers/moves


class DroneState:
    def __init__(self, idx: int, pos: np.ndarray):
        self.idx             = idx
        self.position        = pos.astype(float)
        self.velocity        = (np.random.rand(2) - 0.5) * 2.0
        self.status          = S_ACTIVE
        self.flicker_counter = 0
        self.at_slot         = False

    @property
    def alive(self):
        return self.status in (S_ACTIVE, S_FAILING, S_AT_SLOT)


class SharedSimulation:
    def __init__(self):
        self.drones: list[DroneState] = []
        self.target         = np.array(TARGET_POS, dtype=float)
        self.formation_mode = "circle"   # "circle" | "leader"
        self.step_count     = 0
        self.episode_done   = False
        self.all_at_slots   = False
        self._pending: dict[int, np.ndarray] = {}
        self.collisions: dict[int, bool] = {}
        self._pulse         = 0.0        # comm link animation counter
        self._init_drones()

    def _clear_slot_reward_flags(self):
        for d in self.drones:
            if hasattr(d, '_slot_rewarded'):
                delattr(d, '_slot_rewarded')

    # ── setup ────────────────────────────────────────────────

    def _init_drones(self):
        cx, cy = WIDTH / 2, HEIGHT / 2
        self.drones = []
        for i in range(NUM_DRONES):
            angle = 2 * math.pi * i / NUM_DRONES
            r     = SPAWN_RADIUS * (0.4 + 0.6 * random.random())
            pos   = np.array([
                cx + r * math.cos(angle),
                cy + r * math.sin(angle),
            ])
            self.drones.append(DroneState(i, pos))
        self._pending   = {}
        self.collisions = {i: False for i in range(NUM_DRONES)}

    def reset(self, target_pos=None):
        if target_pos is not None:
            self.target = np.array(target_pos, dtype=float)
        self.step_count   = 0
        self.episode_done = False
        self.all_at_slots = False
        self._init_drones()
        self._clear_slot_reward_flags()

    def set_target(self, pos):
        """User clicked — move target, drones re-route immediately."""
        self.target = np.array(pos, dtype=float)
        # reset at_slot flags so drones fly to new slots
        for d in self.drones:
            if d.status == S_AT_SLOT:
                d.status  = S_ACTIVE
                d.at_slot = False
        self.all_at_slots = False
        self._clear_slot_reward_flags()

    def set_formation(self, mode: str):
        self.formation_mode = mode
        # reset slot flags so drones reform
        for d in self.drones:
            if d.alive:
                d.status  = S_ACTIVE
                d.at_slot = False
        self.all_at_slots = False
        self._clear_slot_reward_flags()

    # ── formation slots ──────────────────────────────────────

    def get_formation_slot(self, drone_idx: int) -> np.ndarray:
        """
        Return the world position this drone should fly to and hover at.

        CIRCLE: evenly-spaced positions on a ring around target.
        LEADER: drone 0 → target itself; others → V-shape behind leader.
        """
        alive_ids = sorted([d.idx for d in self.drones if d.alive])
        # remap to position in alive list for even spacing
        try:
            rank = alive_ids.index(drone_idx)
        except ValueError:
            return self.target.copy()

        n = len(alive_ids)

        if self.formation_mode == "circle":
            angle = 2 * math.pi * rank / max(n, 1)
            offset = FORMATION_CIRCLE_RADIUS * np.array([
                math.cos(angle), math.sin(angle)
            ])
            return self.target + offset

        else:  # leader
            if drone_idx == 0:
                return self.target.copy()
            # V-shape: alternate left/right rows behind leader
            row  = (rank + 1) // 2
            side = 1 if rank % 2 == 1 else -1
            # "behind" = negative X direction from target
            return self.target + np.array([
                -row * FORMATION_V_SPACING,
                side * row * FORMATION_V_SPACING * 0.7,
            ])

    def get_drone_goal(self, drone_idx: int) -> np.ndarray:
        """Return the current destination for a drone."""
        return self.get_formation_slot(drone_idx)

    # ── action queue / sync ──────────────────────────────────

    def push_action(self, idx: int, action: np.ndarray):
        if self.episode_done:
            return
        d = self.drones[idx]
        if d.alive:
            self._pending[idx] = np.array(action, dtype=float)
        # check if all alive drones have submitted
        needed = {d.idx for d in self.drones if d.alive}
        if needed and needed.issubset(self._pending.keys()):
            self._step_world()

    # ── physics ──────────────────────────────────────────────

    def _step_world(self):
        self.collisions = {i: False for i in range(NUM_DRONES)}
        self._pulse     = (self._pulse + 0.06) % (2 * math.pi)

        # 1. apply actions
        for idx, action in self._pending.items():
            d     = self.drones[idx]
            if not d.alive:
                continue
            force        = np.clip(action, -1.0, 1.0) * ACTION_SCALE
            d.velocity  += force
            # hover micro-oscillation so drones never freeze
            d.velocity  += (np.random.rand(2) - 0.5) * 0.08
            speed        = np.linalg.norm(d.velocity)
            if speed > MAX_SPEED:
                d.velocity = d.velocity / speed * MAX_SPEED
            d.position  += d.velocity
            # boundary bounce
            for axis, limit in ((0, WIDTH), (1, HEIGHT)):
                if d.position[axis] < 0 or d.position[axis] > limit:
                    d.velocity[axis] *= -1
            d.position[0] = np.clip(d.position[0], 0, WIDTH)
            d.position[1] = np.clip(d.position[1], 0, HEIGHT)

        self._pending = {}

        # 2. collision detection
        alive = [d for d in self.drones if d.alive]
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                if np.linalg.norm(alive[i].position - alive[j].position) < DANGER_RADIUS:
                    self.collisions[alive[i].idx] = True
                    self.collisions[alive[j].idx] = True

        # 3. at-slot detection (continuous hover — not a terminal state)
        for d in [x for x in self.drones if x.alive]:
            slot = self.get_formation_slot(d.idx)
            d.at_slot = np.linalg.norm(d.position - slot) < TARGET_RADIUS * 0.8
            if d.at_slot and d.status == S_ACTIVE:
                d.status = S_AT_SLOT

        # 4. random failure
        for d in [x for x in self.drones if x.status in (S_ACTIVE, S_AT_SLOT)]:
            if random.random() < FAILURE_PROB_PER_STEP:
                d.status          = S_FAILING
                d.flicker_counter = FAILURE_FLICKER_FRAMES

        # 5. advance FAILING drones
        for d in [x for x in self.drones if x.status == S_FAILING]:
            d.flicker_counter -= 1
            if d.flicker_counter <= 0:
                d.status = S_FAILED

        # 6. all-at-slots check
        alive_drones = [d for d in self.drones if d.alive]
        if alive_drones and all(d.at_slot for d in alive_drones):
            self.all_at_slots = True

        # 7. episode termination
        self.step_count += 1
        n_alive = sum(1 for d in self.drones if d.alive)
        # If MAX_STEPS_PER_EP <= 0, treat as "no time limit" (only failures end episode)
        time_limit_reached = (MAX_STEPS_PER_EP > 0 and self.step_count >= MAX_STEPS_PER_EP)
        if time_limit_reached or n_alive < MIN_DRONES_TO_CONTINUE:
            self.episode_done = True

    # ── helpers ──────────────────────────────────────────────

    def get_drone(self, idx):
        return self.drones[idx]

    def force_fail(self, idx: int):
        d = self.drones[idx]
        if d.status in (S_ACTIVE, S_AT_SLOT):
            d.status          = S_FAILING
            d.flicker_counter = FAILURE_FLICKER_FRAMES

    def comm_pairs(self) -> list[tuple]:
        """Return pairs of (pos_a, pos_b) for drones within COMM_RANGE."""
        alive = [d for d in self.drones if d.alive]
        pairs = []
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                dist = np.linalg.norm(alive[i].position - alive[j].position)
                if dist <= COMM_RANGE:
                    pairs.append((alive[i].position.copy(),
                                  alive[j].position.copy(),
                                  dist))
        return pairs

    def n_alive(self):
        return sum(1 for d in self.drones if d.alive)

    def n_at_slot(self):
        return sum(1 for d in self.drones if d.at_slot and d.alive)
