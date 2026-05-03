# ============================================================
#  config.py  —  All hyperparameters and simulation constants
# ============================================================

# ── Window ───────────────────────────────────────────────────
WIDTH,  HEIGHT = 900, 650
FPS            = 60

# ── Swarm ────────────────────────────────────────────────────
NUM_DRONES          = 10
MAX_SPEED           = 3.5
DRONE_RADIUS        = 8
DANGER_RADIUS       = 22
SPAWN_RADIUS        = 80

# ── Target zone ──────────────────────────────────────────────
TARGET_POS          = (600, 300)   # default — overridden by mouse click in demo
TARGET_RADIUS       = 36

# ── Formation ────────────────────────────────────────────────
FORMATION_CIRCLE_RADIUS = 80    # px — orbit radius in circle mode
FORMATION_V_SPACING     = 38    # px — row/col spacing in leader mode
HOVER_RADIUS            = 10    # px — micro-oscillation around slot
# Keys: C = circle, L = leader  (toggled live in demo)

# ── Communication ─────────────────────────────────────────────
COMM_RANGE          = 200       # px — link drawn when two drones within this
COMM_PULSE_SPEED    = 0.06      # animation pulse speed

# ── Failure ──────────────────────────────────────────────────
FAILURE_PROB_PER_STEP   = 0.00006
FAILURE_FLICKER_FRAMES  = 45
MIN_DRONES_TO_CONTINUE  = 3

# ── Sensor / FOV ─────────────────────────────────────────────
FOV_ANGLE  = 120
FOV_RANGE  = 130

# ── Observation  (total = 22) ─────────────────────────────────
NUM_NEIGHBOURS_OBS = 3
# [vel x2][pos x2][target x3][neighbours 3×4][min_dist][slot rel x2]
OBS_SIZE = 4 + 3 + NUM_NEIGHBOURS_OBS * 4 + 1 + 2   # 22

# ── Action ───────────────────────────────────────────────────
ACTION_SCALE = 0.35

# ── Reward weights ───────────────────────────────────────────
R_TARGET_PROGRESS = 2.5   # reward for moving closer to the global target
R_SLOT_PROGRESS   = 4.0   # reward for moving closer to the assigned formation slot
R_TARGET_CLOSE    = 1.0   # reward for being near the target
R_SLOT_CLOSE      = 3.0   # reward for being near the assigned slot
R_FORMATION       = 2.0   # reward for compact swarm cohesion
R_SEPARATION      = 3.5   # penalty/reward for keeping safe spacing between drones
R_ALIGNMENT       = 0.8   # reward for matching nearby velocity direction
R_COLLISION     = -20.0
R_ALIVE         = +0.05
R_SLOT_REACHED  = +8.0
R_TARGET_REACHED = +12.0
R_ARRIVAL_BONUS = +120.0

# ── PPO ──────────────────────────────────────────────────────
PPO_LEARNING_RATE = 3e-4
PPO_N_STEPS       = 1024
PPO_BATCH_SIZE    = 64
PPO_N_EPOCHS      = 10
PPO_GAMMA         = 0.99
PPO_GAE_LAMBDA    = 0.95
PPO_CLIP_RANGE    = 0.2
PPO_ENT_COEF      = 0.01
POLICY_LAYERS     = [128, 128]

# ── Training ─────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_500_000
CHECKPOINT_FREQ = 100_000
MODELS_DIR      = "models"
LOG_DIR         = "logs"

# ── Episode ──────────────────────────────────────────────────
# Set to a positive integer to enable a per-episode time limit.
# Set to 0 to disable the time limit (run until failure condition).
MAX_STEPS_PER_EP = 0
