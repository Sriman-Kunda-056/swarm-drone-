# ============================================================
#  simulation.py  —  Shared physics engine for all 10 drones
#
#  One SharedSimulation instance is created once and shared
#  by reference across all DroneAgentEnv instances.
#  Physics advances only after ALL drones have submitted
#  their action for the current step.
# ============================================================

import numpy as np
import random
from config import (
    WIDTH, HEIGHT, NUM_DRONES, MAX_SPEED,
    DANGER_RADIUS, SPAWN_RADIUS,
    TARGET_POS, TARGET_RADIUS, ARRIVAL_THRESHOLD,
    FAILURE_PROB_PER_STEP, FAILURE_FLICKER_FRAMES,
    MIN_DRONES_TO_CONTINUE, ACTION_SCALE, MAX_STEPS_PER_EP,
)


# ── Drone states ─────────────────────────────────────────────
S_ACTIVE   = "ACTIVE"
S_FAILING  = "FAILING"    # flickering before removal
S_FAILED   = "FAILED"     # removed from simulation
S_ARRIVED  = "ARRIVED"


class DroneState:
    """Lightweight data object — one per drone."""
    def __init__(self, idx: int, pos: np.ndarray):
        self.idx             = idx
        self.position        = pos.astype(float)
        self.velocity        = (np.random.rand(2) - 0.5) * 2.0
        self.status          = S_ACTIVE
        self.flicker_counter = 0      # counts down during FAILING state
        self.arrived         = False

    @property
    def active(self):
        return self.status in (S_ACTIVE, S_FAILING)

    @property
    def alive(self):
        return self.status != S_FAILED


class SharedSimulation:
    """
    Owns all drone states and advances physics each frame.

    Synchronisation contract
    ────────────────────────
    Each DroneAgentEnv calls push_action(idx, action) for its drone.
    Once all active drones have pushed, step_world() fires automatically.
    """

    def __init__(self):
        self.drones: list[DroneState] = []
        self.target  = np.array(TARGET_POS, dtype=float)
        self.step_count   = 0
        self.episode_done = False
        self.all_arrived  = False

        # pending actions for this tick
        self._pending: dict[int, np.ndarray] = {}
        # collision events this tick {drone_idx: bool}
        self.collisions: dict[int, bool] = {}
        # new arrivals this tick
        self.new_arrivals: set[int] = set()

        self._init_drones()

    # ── setup ────────────────────────────────────────────────

    def _init_drones(self):
        cx, cy = WIDTH / 2, HEIGHT / 2
        self.drones = []
        for i in range(NUM_DRONES):
            angle = 2 * np.pi * i / NUM_DRONES
            r     = SPAWN_RADIUS * np.sqrt(np.random.rand())
            pos   = np.array([
                cx + r * np.cos(angle),
                cy + r * np.sin(angle),
            ])
            self.drones.append(DroneState(i, pos))
        self._pending  = {}
        self.collisions = {i: False for i in range(NUM_DRONES)}
        self.new_arrivals = set()

    def reset(self, target_pos=None):
        if target_pos is not None:
            self.target = np.array(target_pos, dtype=float)
        self.step_count   = 0
        self.episode_done = False
        self.all_arrived  = False
        self._init_drones()

    def set_target(self, pos):
        """Called at runtime when user clicks the demo window."""
        self.target = np.array(pos, dtype=float)
        # un-arrive all arrived drones so they fly to new target
        for d in self.drones:
            if d.status == S_ARRIVED:
                d.status  = S_ACTIVE
                d.arrived = False
        self.all_arrived = False

    # ── action queue ────────────────────────────────────────

    def push_action(self, idx: int, action: np.ndarray):
        """
        Receive one drone's action.  When all active drones have
        pushed, automatically advance the world.
        """
        if self.episode_done:
            return
        d = self.drones[idx]
        if d.alive and d.status != S_ARRIVED:
            self._pending[idx] = action
        self._try_step()

    def _active_indices(self):
        return [d.idx for d in self.drones
                if d.alive and d.status != S_ARRIVED]

    def _try_step(self):
        needed = set(self._active_indices())
        if needed and not needed.issubset(self._pending.keys()):
            return    # still waiting for some drones
        self.step_world()

    # ── physics ─────────────────────────────────────────────

    def step_world(self):
        self.collisions   = {i: False for i in range(NUM_DRONES)}
        self.new_arrivals = set()

        # 1. apply actions
        for idx, action in self._pending.items():
            d = self.drones[idx]
            if d.status not in (S_ACTIVE, S_FAILING):
                continue
            force         = np.clip(action, -1.0, 1.0) * ACTION_SCALE
            d.velocity   += force
            speed         = np.linalg.norm(d.velocity)
            if speed > MAX_SPEED:
                d.velocity = d.velocity / speed * MAX_SPEED
            d.position   += d.velocity

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
                dist = np.linalg.norm(alive[i].position - alive[j].position)
                if dist < DANGER_RADIUS:
                    self.collisions[alive[i].idx] = True
                    self.collisions[alive[j].idx] = True

        # 3. arrival detection
        for d in [x for x in self.drones if x.status == S_ACTIVE]:
            dist = np.linalg.norm(d.position - self.target)
            if dist < ARRIVAL_THRESHOLD:
                d.status  = S_ARRIVED
                d.arrived = True
                self.new_arrivals.add(d.idx)

        # 4. random failure
        for d in [x for x in self.drones if x.status == S_ACTIVE]:
            if random.random() < FAILURE_PROB_PER_STEP:
                d.status          = S_FAILING
                d.flicker_counter = FAILURE_FLICKER_FRAMES

        # 5. advance FAILING drones
        for d in [x for x in self.drones if x.status == S_FAILING]:
            d.flicker_counter -= 1
            if d.flicker_counter <= 0:
                d.status = S_FAILED

        # 6. check all-arrived
        alive_non_arrived = [x for x in self.drones
                             if x.alive and x.status != S_ARRIVED]
        all_arr = all(d.status == S_ARRIVED
                      for d in self.drones if d.alive)
        if all_arr and any(d.status == S_ARRIVED for d in self.drones):
            self.all_arrived = True

        # 7. episode termination
        self.step_count += 1
        n_alive = sum(1 for d in self.drones if d.alive)
        if (self.all_arrived
                or self.step_count >= MAX_STEPS_PER_EP
                or n_alive < MIN_DRONES_TO_CONTINUE):
            self.episode_done = True

    # ── helpers ──────────────────────────────────────────────

    def get_drone(self, idx: int) -> DroneState:
        return self.drones[idx]

    def force_fail(self, idx: int):
        """Manually trigger a drone failure (demo key-press)."""
        d = self.drones[idx]
        if d.status == S_ACTIVE:
            d.status          = S_FAILING
            d.flicker_counter = FAILURE_FLICKER_FRAMES

    def n_alive(self):
        return sum(1 for d in self.drones if d.alive)

    def n_arrived(self):
        return sum(1 for d in self.drones if d.status == S_ARRIVED)
