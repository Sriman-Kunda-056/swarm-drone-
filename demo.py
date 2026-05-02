# ============================================================
#  demo.py  —  Load trained PPO models and run visual demo
#
#  Run:  python demo.py
#
#  Controls
#  ─────────
#  CLICK       move target zone to mouse position
#  F           force-fail a random active drone
#  R           reset episode
#  ESC / Q     quit
#
#  If no trained models are found, falls back to random
#  actions so the window still opens for testing.
# ============================================================

import os
import sys
import glob
import numpy as np
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    WIDTH, HEIGHT, FPS, NUM_DRONES,
    MODELS_DIR, TARGET_POS,
)
from simulation import SharedSimulation, S_FAILED, S_ARRIVED, S_FAILING
from env import DroneAgentEnv
from hud import (
    draw_target, draw_formation_lines, draw_fov_cones,
    draw_drones, draw_hud,
)


# ── load models ──────────────────────────────────────────────

def load_models(sim) -> list:
    """Load the latest checkpoint for each drone, or None (random policy)."""
    models = []
    for i in range(NUM_DRONES):
        # prefer 'final', then latest checkpoint, then None
        final = os.path.join(MODELS_DIR, f"drone_{i}_final.zip")
        if os.path.exists(final):
            env = DummyVecEnv([lambda idx=i: DroneAgentEnv(idx, sim)])
            m   = PPO.load(final, env=env, device="auto")
            models.append(m)
            print(f"  ✅  drone {i:02d} — loaded {final}")
            continue

        pattern  = os.path.join(MODELS_DIR, f"drone_{i}_*.zip")
        ckpts    = sorted(glob.glob(pattern))
        if ckpts:
            env = DummyVecEnv([lambda idx=i: DroneAgentEnv(idx, sim)])
            m   = PPO.load(ckpts[-1], env=env, device="auto")
            models.append(m)
            print(f"  ✅  drone {i:02d} — loaded {ckpts[-1]}")
        else:
            models.append(None)
            print(f"  ⚠️   drone {i:02d} — no model found, using random policy")

    return models


# ── main demo loop ───────────────────────────────────────────

def run_demo():
    pygame.init()
    screen     = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("IAS Swarm — Trained PPO Demo")
    clock      = pygame.time.Clock()
    font       = pygame.font.SysFont(None, 26)
    font_small = pygame.font.SysFont(None, 18)

    # transparent overlay surfaces
    fov_surf  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # ── build shared sim + envs ──────────────────────────────
    sim  = SharedSimulation()
    envs = [DroneAgentEnv(i, sim) for i in range(NUM_DRONES)]

    print("\n🔍  Loading trained models …")
    models = load_models(sim)
    print()

    # ── reset ────────────────────────────────────────────────
    def do_reset(target_pos=None):
        sim.reset(target_pos=target_pos or TARGET_POS)
        obs_list = []
        for env in envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return obs_list

    obs_list      = do_reset()
    reward_history = []
    episode        = 1
    step           = 0
    mode_label     = "PPO Inference"

    # ── loop ─────────────────────────────────────────────────
    running = True
    while running:
        clock.tick(FPS)

        # ── events ───────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs_list = do_reset()
                    reward_history.clear()
                    episode += 1
                    step     = 0
                elif event.key == pygame.K_f:
                    active = [d for d in sim.drones
                              if d.status not in (S_FAILED, S_ARRIVED, S_FAILING)]
                    if active:
                        import random
                        sim.force_fail(random.choice(active).idx)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if 40 < my < HEIGHT - 28:          # not on HUD bars
                    sim.set_target((mx, my))

        # ── step all drones ───────────────────────────────────
        step_reward = 0.0
        for i, (env, model) in enumerate(zip(envs, models)):
            d = sim.get_drone(i)
            if d.status in (S_FAILED, S_ARRIVED):
                continue

            obs = obs_list[i].reshape(1, -1)
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                action    = action[0]
            else:
                action = env.action_space.sample()

            obs_new, rew, terminated, truncated, info = env.step(action)
            obs_list[i] = obs_new
            step_reward += rew

        reward_history.append(step_reward)
        step += 1

        # ── auto-reset on episode end ─────────────────────────
        if sim.episode_done:
            pygame.time.wait(800)    # brief pause to show final state
            obs_list = do_reset()
            reward_history.clear()
            episode += 1
            step     = 0

        # ════════════════════════════════════════════════════
        #  RENDER
        # ════════════════════════════════════════════════════
        screen.fill((8, 10, 18))

        fov_surf.fill((0, 0, 0, 0))
        draw_fov_cones(fov_surf, sim)
        draw_formation_lines(fov_surf, sim)
        draw_target(screen, fov_surf, sim)
        screen.blit(fov_surf, (0, 0))

        draw_drones(screen, font_small, sim)
        draw_hud(screen, font, font_small, sim,
                 mode_label, reward_history, episode, step)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_demo()
