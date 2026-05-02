# ============================================================
#  config.py  —  All hyperparameters and simulation constants
#  Edit this file to tune behaviour without touching any other
# ============================================================

# ── Window ───────────────────────────────────────────────────
WIDTH,  HEIGHT = 800, 600
FPS            = 60

# ── Swarm ────────────────────────────────────────────────────
NUM_DRONES          = 10
MAX_SPEED           = 4.0
DRONE_RADIUS        = 8        # pixels (collision geometry)
DANGER_RADIUS       = 24       # pixels (collision penalty zone)
SPAWN_RADIUS        = 60       # how spread-out drones spawn around centre

# ── Target zone ──────────────────────────────────────────────
# Default target position — overridden at runtime by mouse-click
TARGET_POS          = (600, 300)
TARGET_RADIUS       = 40       # pixels — "arrival" zone
ARRIVAL_THRESHOLD   = TARGET_RADIUS   # drone considered arrived when dist < this

# ── Failure mechanism ────────────────────────────────────────
FAILURE_PROB_PER_STEP   = 0.0008   # random failure chance each step
FAILURE_FLICKER_FRAMES  = 40       # visual warning frames before removal
MIN_DRONES_TO_CONTINUE  = 3        # episode ends if fewer drones remain

# ── Sensor / FOV ─────────────────────────────────────────────
FOV_ANGLE  = 120    # degrees
FOV_RANGE  = 120    # pixels

# ── Observation space ────────────────────────────────────────
NUM_NEIGHBOURS_OBS  = 3        # how many nearest neighbours in obs vector
# Total obs size = 4 (self) + 3 (target) + NUM_NEIGHBOURS_OBS*4 + 1 = 20
OBS_SIZE = 4 + 3 + NUM_NEIGHBOURS_OBS * 4 + 1   # = 20

# ── Action space ─────────────────────────────────────────────
ACTION_SCALE = 0.4   # multiplies raw [-1,1] action → force applied

# ── Reward weights ───────────────────────────────────────────
R_APPROACH        = -0.8    # per-step: scales −dist_to_target
R_FORMATION       = -0.3    # per-step: scales −std of dists to centroid
R_COLLISION       = -15.0   # one-off per collision event
R_ALIVE           = +0.05   # per-step survival bonus
R_ARRIVAL_BONUS   = +150.0  # sparse: entire swarm reaches target
R_PARTIAL_ARRIVAL = +10.0   # bonus when THIS drone reaches target

# ── PPO hyperparameters ──────────────────────────────────────
PPO_LEARNING_RATE   = 3e-4
PPO_N_STEPS         = 1024
PPO_BATCH_SIZE      = 64
PPO_N_EPOCHS        = 10
PPO_GAMMA           = 0.99
PPO_GAE_LAMBDA      = 0.95
PPO_CLIP_RANGE      = 0.2
PPO_ENT_COEF        = 0.01
POLICY_LAYERS       = [128, 128]

# ── Training ─────────────────────────────────────────────────
TOTAL_TIMESTEPS     = 1_500_000
CHECKPOINT_FREQ     = 100_000
MODELS_DIR          = "models"
LOG_DIR             = "logs"

# ── Episode ──────────────────────────────────────────────────
MAX_STEPS_PER_EP    = 800
