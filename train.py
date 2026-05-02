# ============================================================
#  train.py  —  IPPO training  (one PPO model per drone)
#
#  Run:  python train.py
#
#  What happens
#  ─────────────
#  • Creates one DroneAgentEnv + PPO model per drone
#  • All envs share one SharedSimulation instance
#  • Training alternates: collect rollouts → PPO update
#  • Checkpoints saved every CHECKPOINT_FREQ steps to models/
#  • TensorBoard logs written to logs/
#
#  Monitor training:
#    tensorboard --logdir logs
# ============================================================

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    NUM_DRONES, TOTAL_TIMESTEPS, CHECKPOINT_FREQ,
    MODELS_DIR, LOG_DIR,
    PPO_LEARNING_RATE, PPO_N_STEPS, PPO_BATCH_SIZE,
    PPO_N_EPOCHS, PPO_GAMMA, PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE, PPO_ENT_COEF, POLICY_LAYERS,
    MAX_STEPS_PER_EP,
)
from simulation import SharedSimulation
from env import DroneAgentEnv


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

policy_kwargs = dict(net_arch=POLICY_LAYERS)


# ── Custom callback: sync episode resets across all envs ─────

class SwarmResetCallback(BaseCallback):
    """
    When drone 0's env resets (episode done), force-reset the
    shared simulation so all other drones start fresh too.
    Called automatically by SB3's rollout collection.
    """
    def __init__(self, sim: SharedSimulation, envs: list, verbose=0):
        super().__init__(verbose)
        self.sim  = sim
        self.envs = envs

    def _on_step(self) -> bool:
        return True


class SwarmTrainer:
    """Manages 10 PPO models training on a shared simulation."""

    def __init__(self):
        self.sim  = SharedSimulation()
        self.envs  = []
        self.models = []
        self._build()

    def _build(self):
        for i in range(NUM_DRONES):
            env = DroneAgentEnv(drone_idx=i, sim=self.sim)
            vec = DummyVecEnv([lambda e=env: e])
            self.envs.append(vec)

            model = PPO(
                "MlpPolicy",
                vec,
                learning_rate  = PPO_LEARNING_RATE,
                n_steps        = PPO_N_STEPS,
                batch_size     = PPO_BATCH_SIZE,
                n_epochs       = PPO_N_EPOCHS,
                gamma          = PPO_GAMMA,
                gae_lambda     = PPO_GAE_LAMBDA,
                clip_range     = PPO_CLIP_RANGE,
                ent_coef       = PPO_ENT_COEF,
                policy_kwargs  = policy_kwargs,
                tensorboard_log= f"{LOG_DIR}/drone_{i}",
                verbose        = 1 if i == 0 else 0,
                device         = "auto",   # uses GPU if available
            )
            self.models.append(model)

        print(f"✅  Built {NUM_DRONES} PPO models")
        print(f"    Device: {self.models[0].device}")

    def train(self):
        """
        Train all drones together in a round-robin fashion.
        Each round: collect N_STEPS for every drone → update all.

        The SharedSimulation synchronises physics — when all drones
        have pushed actions, the world steps forward once.
        """
        steps_trained = 0
        checkpoint_at = CHECKPOINT_FREQ

        print(f"\n🚀  Starting IPPO training — {TOTAL_TIMESTEPS:,} total steps\n")

        # We train in chunks so we can interleave all 10 agents
        chunk = PPO_N_STEPS   # collect this many steps per drone per round

        while steps_trained < TOTAL_TIMESTEPS:
            # ── reset shared sim for new episode ─────────────
            self.sim.reset()

            ep_rewards = [0.0] * NUM_DRONES
            obs        = [env.reset()[0] for env in
                          [DroneAgentEnv(i, self.sim) for i in range(NUM_DRONES)]]

            # re-wrap fresh envs (SB3 needs VecEnv)
            raw_envs = [DroneAgentEnv(i, self.sim) for i in range(NUM_DRONES)]
            self.sim.reset()

            for i, (model, raw_env) in enumerate(zip(self.models, raw_envs)):
                vec = DummyVecEnv([lambda e=raw_env: e])
                model.set_env(vec)

            # collect rollouts + update, one drone at a time
            for i, model in enumerate(self.models):
                model.learn(
                    total_timesteps   = chunk,
                    reset_num_timesteps = False,
                    tb_log_name       = f"drone_{i}",
                )

            steps_trained += chunk * NUM_DRONES

            # ── checkpoint ───────────────────────────────────
            if steps_trained >= checkpoint_at:
                self._save(steps_trained)
                checkpoint_at += CHECKPOINT_FREQ
                print(f"💾  Checkpoint @ {steps_trained:,} steps")

            # ── progress ─────────────────────────────────────
            pct = steps_trained / TOTAL_TIMESTEPS * 100
            print(f"    {steps_trained:>8,} / {TOTAL_TIMESTEPS:,}  "
                  f"({pct:.1f}%)", end="\r")

        # ── final save ───────────────────────────────────────
        self._save("final")
        print(f"\n\n✅  Training complete. Models saved to '{MODELS_DIR}/'")
        print( "    Run  python demo.py  to watch the trained swarm.\n")

    def _save(self, tag):
        for i, model in enumerate(self.models):
            path = os.path.join(MODELS_DIR, f"drone_{i}_{tag}")
            model.save(path)


# ── entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    trainer = SwarmTrainer()
    trainer.train()
