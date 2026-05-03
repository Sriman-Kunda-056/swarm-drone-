# ============================================================
#  train.py  —  IPPO training (10 PPO models, obs size 22)
#  Run:  python train.py
#  Monitor: tensorboard --logdir logs
# ============================================================

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import (
    NUM_DRONES, TOTAL_TIMESTEPS, CHECKPOINT_FREQ,
    MODELS_DIR, LOG_DIR,
    PPO_LEARNING_RATE, PPO_N_STEPS, PPO_BATCH_SIZE,
    PPO_N_EPOCHS, PPO_GAMMA, PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE, PPO_ENT_COEF, POLICY_LAYERS,
)
from simulation import SharedSimulation
from env import DroneAgentEnv

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

policy_kwargs = dict(net_arch=POLICY_LAYERS)


class SwarmTrainer:
    def __init__(self):
        self.sim    = SharedSimulation()
        self.models = []
        self._build()

    def _build(self):
        for i in range(NUM_DRONES):
            env   = DroneAgentEnv(drone_idx=i, sim=self.sim)
            vec   = DummyVecEnv([lambda e=env: e])
            model = PPO(
                "MlpPolicy", vec,
                learning_rate   = PPO_LEARNING_RATE,
                n_steps         = PPO_N_STEPS,
                batch_size      = PPO_BATCH_SIZE,
                n_epochs        = PPO_N_EPOCHS,
                gamma           = PPO_GAMMA,
                gae_lambda      = PPO_GAE_LAMBDA,
                clip_range      = PPO_CLIP_RANGE,
                ent_coef        = PPO_ENT_COEF,
                policy_kwargs   = policy_kwargs,
                tensorboard_log = f"{LOG_DIR}/drone_{i}",
                verbose         = 1 if i == 0 else 0,
                device          = "auto",
            )
            self.models.append(model)
        print(f"✅  {NUM_DRONES} PPO models  |  device: {self.models[0].device}")

    def train(self):
        steps = 0
        ckpt_at = CHECKPOINT_FREQ
        chunk   = PPO_N_STEPS
        print(f"\n🚀  Training — {TOTAL_TIMESTEPS:,} steps\n")

        while steps < TOTAL_TIMESTEPS:
            self.sim.reset()
            raw_envs = [DroneAgentEnv(i, self.sim) for i in range(NUM_DRONES)]
            self.sim.reset()

            for i, (model, raw_env) in enumerate(zip(self.models, raw_envs)):
                model.set_env(DummyVecEnv([lambda e=raw_env: e]))
                model.learn(total_timesteps=chunk,
                            reset_num_timesteps=False,
                            tb_log_name=f"drone_{i}")

            steps += chunk * NUM_DRONES

            if steps >= ckpt_at:
                self._save(steps)
                print(f"\n💾  Checkpoint @ {steps:,}")
                ckpt_at += CHECKPOINT_FREQ

            pct = steps / TOTAL_TIMESTEPS * 100
            print(f"    {steps:>9,} / {TOTAL_TIMESTEPS:,}  ({pct:.1f}%)", end="\r")

        self._save("final")
        print(f"\n✅  Done. Run  python demo.py  to watch.\n")

    def _save(self, tag):
        for i, m in enumerate(self.models):
            m.save(os.path.join(MODELS_DIR, f"drone_{i}_{tag}"))


if __name__ == "__main__":
    SwarmTrainer().train()
