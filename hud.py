# ============================================================
#  hud.py  —  All pygame rendering for demo.py
# ============================================================

import math
import random
import pygame
import numpy as np
from config import (
    WIDTH, HEIGHT, FOV_ANGLE, FOV_RANGE,
    TARGET_RADIUS, DANGER_RADIUS,
)
from simulation import S_ACTIVE, S_FAILING, S_FAILED, S_ARRIVED


# ── colours ─────────────────────────────────────────────────
C_BG          = (8,  10,  18)
C_TARGET_FILL = (255, 220, 60,  30)
C_TARGET_RING = (255, 220, 60)
C_LEADER      = (255, 220,  0)
C_FLOCKING    = (60,  180, 255)
C_ARRIVED     = (80,  255, 140)
C_FAILING     = (255, 140,  40)
C_FAILED      = (255,  50,  50)
C_FORMATION   = (60,  255, 180,  18)
C_COMM        = (0,   255, 140,  30)
C_HUD_BG      = (14,  16,  24)
C_HUD_LINE    = (40,  44,  60)


def _drone_color(drone):
    if drone.status == S_FAILED:
        return C_FAILED
    if drone.status == S_FAILING:
        return C_FAILING
    if drone.status == S_ARRIVED:
        return C_ARRIVED
    if drone.idx == 0:
        return C_LEADER
    speed = np.linalg.norm(drone.velocity)
    i     = min(255, int(speed * 60))
    return (0, i, 255)


def _cone_pts(drone):
    v   = drone.velocity
    spd = np.linalg.norm(v)
    hdg = math.atan2(v[1], v[0]) if spd > 1e-4 else 0.0
    h   = math.radians(FOV_ANGLE / 2)
    tip = drone.position
    L   = tip + FOV_RANGE * np.array([math.cos(hdg - h), math.sin(hdg - h)])
    R   = tip + FOV_RANGE * np.array([math.cos(hdg + h), math.sin(hdg + h)])
    return [tuple(tip.astype(int)), tuple(L.astype(int)), tuple(R.astype(int))]


# ── spark particles for FAILING animation ────────────────────

class SparkSystem:
    def __init__(self):
        self.particles = []   # [x, y, vx, vy, life]

    def emit(self, pos):
        for _ in range(4):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.particles.append([
                pos[0], pos[1],
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                random.randint(10, 25),
            ])

    def update_draw(self, screen):
        next_p = []
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1
            if p[4] > 0:
                alpha = int(255 * p[4] / 25)
                color = (255, max(0, 100 - p[4] * 4), 0)
                pygame.draw.circle(screen, color,
                                   (int(p[0]), int(p[1])), 2)
                next_p.append(p)
        self.particles = next_p


sparks = SparkSystem()


# ── draw functions ───────────────────────────────────────────

def draw_target(screen, fov_surf, sim):
    """Glowing target zone."""
    tx, ty = int(sim.target[0]), int(sim.target[1])
    # filled semi-transparent circle on fov surface
    pygame.draw.circle(fov_surf, C_TARGET_FILL, (tx, ty), TARGET_RADIUS)
    # outer ring
    pygame.draw.circle(screen, C_TARGET_RING, (tx, ty), TARGET_RADIUS, 2)
    pygame.draw.circle(screen, C_TARGET_RING, (tx, ty), TARGET_RADIUS + 6, 1)
    # crosshair
    pygame.draw.line(screen, C_TARGET_RING, (tx - 12, ty), (tx + 12, ty), 1)
    pygame.draw.line(screen, C_TARGET_RING, (tx, ty - 12), (tx, ty + 12), 1)


def draw_formation_lines(fov_surf, sim):
    """Faint lines connecting active drones to show swarm structure."""
    alive = [d for d in sim.drones if d.alive and d.status != S_FAILED]
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            dist = np.linalg.norm(alive[i].position - alive[j].position)
            if dist < FOV_RANGE * 1.2:
                pygame.draw.line(
                    fov_surf, C_FORMATION,
                    tuple(alive[i].position.astype(int)),
                    tuple(alive[j].position.astype(int)), 1,
                )


def draw_fov_cones(fov_surf, sim):
    for d in sim.drones:
        if d.status in (S_ACTIVE,):
            pts  = _cone_pts(d)
            col  = (60, 160, 255, 14) if d.idx != 0 else (255, 220, 0, 14)
            pygame.draw.polygon(fov_surf, col, pts)


def draw_drones(screen, font_small, sim):
    for d in sim.drones:
        pos = tuple(d.position.astype(int))
        col = _drone_color(d)

        if d.status == S_FAILED:
            pygame.draw.circle(screen, C_FAILED, pos, 6, 1)
            continue

        if d.status == S_FAILING:
            sparks.emit(d.position)

        pygame.draw.circle(screen, col, pos, 7)
        # direction arrow
        end = (d.position + d.velocity * 5).astype(int)
        pygame.draw.line(screen, (255, 255, 255), pos, tuple(end), 2)

        # state label
        labels = {S_ACTIVE: "A", S_FAILING: "!", S_ARRIVED: "✓"}
        lbl = labels.get(d.status, "")
        surf = font_small.render(lbl, True, col)
        screen.blit(surf, (pos[0] + 9, pos[1] - 9))

    sparks.update_draw(screen)


def draw_reward_graph(screen, font_small, reward_history: list):
    """Mini line graph of recent total reward in top-right."""
    if len(reward_history) < 2:
        return
    gx, gy, gw, gh = WIDTH - 170, 48, 160, 60
    pygame.draw.rect(screen, C_HUD_BG, (gx, gy, gw, gh))
    pygame.draw.rect(screen, C_HUD_LINE, (gx, gy, gw, gh), 1)
    label = font_small.render("Σ reward", True, (150, 150, 150))
    screen.blit(label, (gx + 4, gy + 2))

    recent = reward_history[-gw:]
    if not recent:
        return
    mn, mx = min(recent), max(recent)
    rng    = mx - mn if mx != mn else 1.0
    pts = []
    for i, r in enumerate(recent):
        px = gx + int(i * gw / len(recent))
        py = gy + gh - int((r - mn) / rng * (gh - 14)) - 4
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(screen, (80, 255, 140), False, pts, 1)


def draw_hud(screen, font, font_small, sim, mode_label,
             reward_history, episode, step):
    """Top bar, bottom bar, legend, reward graph."""

    n_alive   = sim.n_alive()
    n_arr     = sim.n_arrived()
    n_total   = len(sim.drones)

    # ── top bar ──────────────────────────────────────────────
    pygame.draw.rect(screen, C_HUD_BG,  (0, 0, WIDTH, 40))
    pygame.draw.line(screen, C_HUD_LINE, (0, 40), (WIDTH, 40), 1)
    top = (f"Episode {episode}   Step {step}   "
           f"Alive: {n_alive}/{n_total}   "
           f"Arrived: {n_arr}/{n_total}   "
           f"Mode: {mode_label}")
    screen.blit(font.render(top, True, (220, 220, 230)), (10, 11))

    # ── reward graph ─────────────────────────────────────────
    draw_reward_graph(screen, font_small, reward_history)

    # ── bottom bar ───────────────────────────────────────────
    pygame.draw.rect(screen, C_HUD_BG,  (0, HEIGHT - 28, WIDTH, 28))
    pygame.draw.line(screen, C_HUD_LINE, (0, HEIGHT - 28), (WIDTH, HEIGHT - 28), 1)
    ctrl = ("CLICK: move target   F: force fail   R: reset episode   "
            "ESC: quit")
    screen.blit(font_small.render(ctrl, True, (110, 110, 120)), (10, HEIGHT - 20))

    # ── legend ───────────────────────────────────────────────
    legend = [
        (C_LEADER,  "Leader drone"),
        (C_FLOCKING,"Active drone"),
        (C_ARRIVED, "Arrived ✓"),
        (C_FAILING, "Failing !"),
        (C_FAILED,  "Failed ✕"),
        (C_TARGET_RING, "Target zone"),
    ]
    y = 50
    for col, txt in legend:
        pygame.draw.circle(screen, col, (WIDTH - 115, y), 5)
        screen.blit(font_small.render(txt, True, col), (WIDTH - 105, y - 7))
        y += 20
