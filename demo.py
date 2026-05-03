# ============================================================
#  demo.py  —  Visual demo with smart steering fallback
#
#  Works WITHOUT trained models — smart rule-based steering
#  drives the drones when no .zip is found in models/.
#
#  Controls
#  ─────────
#  CLICK    move target zone
#  C        circle formation
#  L        leader / V-shape formation
#  F        force-fail a random drone
#  R        reset episode
#  ESC/Q    quit
# ============================================================

import os, sys, glob, random, math
import numpy as np
import pygame

from config import (
    WIDTH, HEIGHT, FPS, NUM_DRONES,
    MODELS_DIR, TARGET_POS, ACTION_SCALE,
    DANGER_RADIUS, FOV_RANGE, MAX_SPEED,
)
from simulation import SharedSimulation, S_FAILED, S_FAILING, S_AT_SLOT
from env import DroneAgentEnv
from hud import (
    draw_target, draw_slot_markers, draw_comm_links,
    draw_formation_lines, draw_fov_cones,
    draw_drones, draw_hud,
)


# ════════════════════════════════════════════════════════════
#  Smart fallback steering policy
#  Used when no trained PPO model is available.
#  Produces realistic formation flight without any training.
# ════════════════════════════════════════════════════════════

def smart_action(drone_idx: int, sim: SharedSimulation) -> np.ndarray:
    """
    Rule-based action that steers toward the drone's formation slot
    while maintaining separation from neighbours.

    Returns normalised [ax, ay] ∈ [-1, 1] — same interface as PPO.
    """
    d    = sim.get_drone(drone_idx)
    slot = sim.get_formation_slot(drone_idx)

    # ── 1. slot attraction ───────────────────────────────────
    diff_slot = slot - d.position
    dist_slot = np.linalg.norm(diff_slot)

    if dist_slot > 1e-3:
        # strong pull when far, gentle when close (hover zone)
        strength   = min(1.0, dist_slot / 120.0)
        slot_force = (diff_slot / dist_slot) * strength
    else:
        slot_force = np.zeros(2)

    # ── 2. separation from neighbours ────────────────────────
    sep = np.zeros(2)
    for other in sim.drones:
        if other.idx == drone_idx or not other.alive:
            continue
        diff = d.position - other.position
        dist = np.linalg.norm(diff)
        if dist < DANGER_RADIUS * 2.0 and dist > 1e-3:
            sep += (diff / dist) * (1.0 - dist / (DANGER_RADIUS * 2.0))

    # ── 3. velocity damping (reduces oscillation) ────────────
    damping = -d.velocity * 0.15

    # ── 4. alignment with nearest neighbours (pack feeling) ──
    alignment = np.zeros(2)
    n_count   = 0
    for other in sim.drones:
        if other.idx == drone_idx or not other.alive:
            continue
        dist = np.linalg.norm(d.position - other.position)
        if dist < FOV_RANGE:
            alignment += other.velocity
            n_count   += 1
    if n_count > 0:
        alignment = (alignment / n_count - d.velocity) * 0.08

    # ── combine ──────────────────────────────────────────────
    action = slot_force * 1.0 + sep * 0.9 + damping + alignment
    return np.clip(action / ACTION_SCALE, -1.0, 1.0).astype(np.float32)


def formation_assist(drone_idx: int, sim: SharedSimulation) -> np.ndarray:
    """Small stabilizer used on top of PPO actions to keep the formation tight."""
    d    = sim.get_drone(drone_idx)
    slot = sim.get_formation_slot(drone_idx)

    diff_slot = slot - d.position
    dist_slot = np.linalg.norm(diff_slot)
    if dist_slot > 1e-3:
        slot_force = (diff_slot / dist_slot) * min(0.7, dist_slot / 180.0)
    else:
        slot_force = np.zeros(2)

    sep = np.zeros(2)
    for other in sim.drones:
        if other.idx == drone_idx or not other.alive:
            continue
        diff = d.position - other.position
        dist = np.linalg.norm(diff)
        if dist < DANGER_RADIUS * 1.6 and dist > 1e-3:
            sep += (diff / dist) * (1.0 - dist / (DANGER_RADIUS * 1.6))

    damping = -d.velocity * 0.10
    action = slot_force * 0.9 + sep * 0.55 + damping
    return np.clip(action / ACTION_SCALE, -1.0, 1.0).astype(np.float32)


# ════════════════════════════════════════════════════════════
#  Model loader
# ════════════════════════════════════════════════════════════

def load_models(sim) -> list:
    """Try loading trained PPO models; fall back to None (→ smart_action)."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        sb3_ok = True
    except ImportError:
        sb3_ok = False

    models = []
    for i in range(NUM_DRONES):
        loaded = None
        if sb3_ok:
            final   = os.path.join(MODELS_DIR, f"drone_{1}_final.zip")
            pattern = os.path.join(MODELS_DIR, f"drone_{1}_*.zip")
            ckpts   = glob.glob(pattern)

            def _step_num(path: str) -> int:
                stem = os.path.splitext(os.path.basename(path))[0]
                try:
                    return int(stem.rsplit("_", 1)[-1])
                except ValueError:
                    return -1

            latest = max(ckpts, key=_step_num) if ckpts else None
            path   = final if os.path.exists(final) else latest
            if path:
                try:
                    env = DummyVecEnv([lambda idx=i: DroneAgentEnv(idx, sim)])
                    loaded = PPO.load(path, env=env, device="auto")
                    print(f"  ✅  drone {i:02d} — {path}")
                except Exception as e:
                    print(f"  ⚠️   drone {i:02d} — load failed ({e}), using smart policy")
        models.append(loaded)

    n_smart = models.count(None)
    if n_smart == NUM_DRONES:
        print("  ℹ️   No trained models found — running smart steering policy")
    elif n_smart > 0:
        print(f"  ℹ️   {n_smart} drones using smart policy fallback")
    return models


# ════════════════════════════════════════════════════════════
#  Demo loop
# ════════════════════════════════════════════════════════════

def run_demo():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("IAS Swarm — Formation Demo")
    clock      = pygame.time.Clock()
    font       = pygame.font.SysFont(None, 26)
    font_small = pygame.font.SysFont(None, 18)

    fov_surf  = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    comm_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    form_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    sim  = SharedSimulation()
    envs = [DroneAgentEnv(i, sim) for i in range(NUM_DRONES)]

    print("\n🔍  Loading models …")
    models = load_models(sim)
    print()

    def do_reset(tgt=None):
        sim.reset(target_pos=tgt or TARGET_POS)
        return [env._get_obs() for env in envs]

    obs_list       = do_reset()
    reward_history = []
    episode        = 1
    step           = 0
    paused         = False

    running = True
    while running:
        clock.tick(FPS)

        # ── events ───────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_r:
                    obs_list = do_reset()
                    reward_history.clear()
                    episode += 1; step = 0
                    paused = False
                elif k == pygame.K_c:
                    sim.set_formation("circle")
                elif k == pygame.K_l:
                    sim.set_formation("leader")
                elif k == pygame.K_f:
                    candidates = [d for d in sim.drones
                                  if d.status not in (S_FAILED, S_FAILING)]
                    if candidates:
                        sim.force_fail(random.choice(candidates).idx)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if 40 < my < HEIGHT - 28:
                    sim.set_target((mx, my))

        # ── step all drones (skip when paused at episode end) ───
        if not paused:
            step_reward = 0.0
            for i, (env, model) in enumerate(zip(envs, models)):
                d = sim.get_drone(i)
                if d.status == S_FAILED:
                    continue

                if model is not None:
                    obs    = obs_list[i].reshape(1, -1)
                    action, _ = model.predict(obs, deterministic=True)
                    action = action[0]
                    action = np.clip(
                        action * 0.82 + formation_assist(i, sim) * 0.28,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                else:
                    action = smart_action(i, sim)

                obs_new, rew, terminated, _, info = env.step(action)
                obs_list[i] = obs_new
                step_reward += rew

            reward_history.append(step_reward)
            step += 1

            if sim.episode_done:
                paused = True

        # ── render ───────────────────────────────────────────
        screen.fill((8, 10, 18))

        # FOV cones
        fov_surf.fill((0, 0, 0, 0))
        draw_fov_cones(fov_surf, sim)
        screen.blit(fov_surf, (0, 0))

        # formation lines + comm links
        form_surf.fill((0, 0, 0, 0))
        draw_formation_lines(form_surf, sim)
        screen.blit(form_surf, (0, 0))

        comm_surf.fill((0, 0, 0, 0))
        draw_comm_links(comm_surf, sim)
        screen.blit(comm_surf, (0, 0))

        # target + slot markers
        draw_target(screen, fov_surf, sim)
        screen.blit(fov_surf, (0, 0))
        draw_slot_markers(screen, sim)

        # drones
        draw_drones(screen, font_small, sim)

        # HUD
        draw_hud(screen, font, font_small, sim,
                 sim.formation_mode, reward_history, episode, step)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_demo()
