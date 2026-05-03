# ============================================================
#  hud.py  —  All pygame rendering
# ============================================================

import math
import random
import pygame
import numpy as np
from config import (
    WIDTH, HEIGHT, FOV_ANGLE, FOV_RANGE,
    TARGET_RADIUS, DANGER_RADIUS, COMM_RANGE,
    FORMATION_CIRCLE_RADIUS,
)
from simulation import S_ACTIVE, S_FAILING, S_FAILED, S_AT_SLOT

# ── palette ──────────────────────────────────────────────────
C_BG         = ( 8,  10,  18)
C_HUD_BG     = (14,  16,  26)
C_HUD_LINE   = (38,  42,  60)
C_LEADER     = (255, 220,   0)
C_ACTIVE     = ( 60, 180, 255)
C_AT_SLOT    = ( 80, 255, 140)
C_FAILING    = (255, 140,  40)
C_FAILED     = (120,  40,  40)
C_TARGET     = (255, 220,  60)
C_COMM       = (  0, 200, 120)
C_SLOT_MKR   = (255, 255, 255)
C_FORM_LINE  = ( 40, 180, 120)


def _drone_color(drone):
    if drone.status == S_FAILED:  return C_FAILED
    if drone.status == S_FAILING: return C_FAILING
    if drone.status == S_AT_SLOT: return C_AT_SLOT
    if drone.idx == 0:            return C_LEADER
    spd = np.linalg.norm(drone.velocity)
    i   = min(255, int(spd * 60))
    return (0, i, 255)


# ── spark system ─────────────────────────────────────────────

class SparkSystem:
    def __init__(self):
        self.particles = []

    def emit(self, pos):
        for _ in range(5):
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(1.0, 3.5)
            self.particles.append([
                float(pos[0]), float(pos[1]),
                math.cos(a) * s, math.sin(a) * s,
                random.randint(12, 28),
            ])

    def update_draw(self, screen):
        keep = []
        for p in self.particles:
            p[0] += p[2]; p[1] += p[3]; p[4] -= 1
            if p[4] > 0:
                c = (255, max(0, int(80 - p[4] * 3)), 0)
                pygame.draw.circle(screen, c, (int(p[0]), int(p[1])), 2)
                keep.append(p)
        self.particles = keep


sparks = SparkSystem()


# ── draw helpers ─────────────────────────────────────────────

def _cone_pts(drone):
    v   = drone.velocity
    spd = np.linalg.norm(v)
    hdg = math.atan2(v[1], v[0]) if spd > 1e-4 else 0.0
    h   = math.radians(FOV_ANGLE / 2)
    tip = drone.position
    L   = tip + FOV_RANGE * np.array([math.cos(hdg - h), math.sin(hdg - h)])
    R   = tip + FOV_RANGE * np.array([math.cos(hdg + h), math.sin(hdg + h)])
    return [tuple(tip.astype(int)), tuple(L.astype(int)), tuple(R.astype(int))]


def draw_target(screen, fov_surf, sim):
    tx, ty = int(sim.target[0]), int(sim.target[1])
    pygame.draw.circle(fov_surf, (*C_TARGET, 22), (tx, ty), TARGET_RADIUS)
    pygame.draw.circle(screen, C_TARGET, (tx, ty), TARGET_RADIUS, 2)
    pygame.draw.circle(screen, (*C_TARGET[:3], 80), (tx, ty), TARGET_RADIUS + 8, 1)
    pygame.draw.line(screen, C_TARGET, (tx-14, ty), (tx+14, ty), 1)
    pygame.draw.line(screen, C_TARGET, (tx, ty-14), (tx, ty+14), 1)


def draw_slot_markers(screen, sim):
    """Small X markers showing each drone's formation slot."""
    for d in sim.drones:
        if not d.alive:
            continue
        slot = sim.get_formation_slot(d.idx)
        sx, sy = int(slot[0]), int(slot[1])
        col = (*_drone_color(d)[:3], 120)
        sz  = 5
        pygame.draw.line(screen, _drone_color(d),
                         (sx-sz, sy-sz), (sx+sz, sy+sz), 1)
        pygame.draw.line(screen, _drone_color(d),
                         (sx+sz, sy-sz), (sx-sz, sy+sz), 1)


def draw_comm_links(overlay, sim):
    """
    Draw pulsing lines between every pair of drones within COMM_RANGE.
    Opacity scales with distance — closer = brighter.
    """
    pairs = sim.comm_pairs()
    pulse = abs(math.sin(sim._pulse))
    for pa, pb, dist in pairs:
        # fade with distance
        alpha = int((1 - dist / COMM_RANGE) * 80 * (0.4 + 0.6 * pulse))
        alpha = max(10, min(100, alpha))
        pygame.draw.line(overlay, (*C_COMM, alpha),
                         tuple(pa.astype(int)),
                         tuple(pb.astype(int)), 1)


def draw_formation_lines(overlay, sim):
    """Connect drones that are in the same pack (near each other)."""
    alive = [d for d in sim.drones if d.alive]
    for i in range(len(alive)):
        for j in range(i+1, len(alive)):
            dist = np.linalg.norm(alive[i].position - alive[j].position)
            if dist < FOV_RANGE * 1.1:
                alpha = max(8, int(30 * (1 - dist / (FOV_RANGE * 1.1))))
                pygame.draw.line(overlay, (*C_FORM_LINE, alpha),
                                 tuple(alive[i].position.astype(int)),
                                 tuple(alive[j].position.astype(int)), 1)


def draw_fov_cones(fov_surf, sim):
    for d in sim.drones:
        if d.status in (S_ACTIVE, S_AT_SLOT):
            pts = _cone_pts(d)
            col = (255, 220, 0, 10) if d.idx == 0 else (60, 160, 255, 10)
            pygame.draw.polygon(fov_surf, col, pts)


def draw_drones(screen, font_small, sim):
    for d in sim.drones:
        pos = tuple(d.position.astype(int))
        col = _drone_color(d)

        if d.status == S_FAILED:
            pygame.draw.circle(screen, C_FAILED, pos, 5, 1)
            continue

        if d.status == S_FAILING:
            sparks.emit(d.position)

        # body
        pygame.draw.circle(screen, col, pos, 7)
        # outline ring for leader
        if d.idx == 0:
            pygame.draw.circle(screen, (255, 255, 200), pos, 10, 1)

        # direction arrow
        end = (d.position + d.velocity * 5).astype(int)
        pygame.draw.line(screen, (220, 220, 220), pos, tuple(end), 2)

        # state label
        lbl_map = {S_ACTIVE: "", S_AT_SLOT: "✓", S_FAILING: "!"}
        lbl_txt = lbl_map.get(d.status, "")
        if lbl_txt:
            surf = font_small.render(lbl_txt, True, col)
            screen.blit(surf, (pos[0]+9, pos[1]-9))

    sparks.update_draw(screen)


def draw_reward_graph(screen, font_small, history):
    if len(history) < 2:
        return
    gx, gy, gw, gh = WIDTH - 175, 48, 165, 55
    pygame.draw.rect(screen, C_HUD_BG,  (gx, gy, gw, gh))
    pygame.draw.rect(screen, C_HUD_LINE,(gx, gy, gw, gh), 1)
    screen.blit(font_small.render("Σ reward / step", True, (130,130,130)),
                (gx+4, gy+2))
    recent = history[-(gw):]
    if len(recent) < 2:
        return
    mn, mx = min(recent), max(recent)
    rng = mx - mn if mx != mn else 1.0
    pts = []
    for i, r in enumerate(recent):
        px = gx + int(i * gw / len(recent))
        py = gy + gh - int((r - mn) / rng * (gh-14)) - 4
        pts.append((px, py))
    pygame.draw.lines(screen, C_AT_SLOT, False, pts, 1)


def draw_hud(screen, font, font_small, sim, formation_mode,
             reward_history, episode, step):
    n_alive = sim.n_alive()
    n_slot  = sim.n_at_slot()
    n_total = NUM_DRONES = len(sim.drones)

    # top bar
    pygame.draw.rect(screen, C_HUD_BG,  (0, 0, WIDTH, 40))
    pygame.draw.line(screen, C_HUD_LINE, (0, 40), (WIDTH, 40), 1)
    top = (f"Ep {episode}  Step {step}  "
           f"Alive: {n_alive}/{n_total}  "
           f"In Formation: {n_slot}/{n_total}  "
           f"Formation: {formation_mode.upper()}")
    screen.blit(font.render(top, True, (220, 220, 235)), (10, 11))

    draw_reward_graph(screen, font_small, reward_history)

    # bottom bar
    pygame.draw.rect(screen, C_HUD_BG,  (0, HEIGHT-28, WIDTH, 28))
    pygame.draw.line(screen, C_HUD_LINE, (0, HEIGHT-28), (WIDTH, HEIGHT-28), 1)
    ctrl = ("CLICK: move target   C: circle formation   L: leader formation   "
            "F: force fail   R: reset   ESC: quit")
    screen.blit(font_small.render(ctrl, True, (100,100,115)), (8, HEIGHT-20))

    # legend
    items = [
        (C_LEADER,  "Leader (idx 0)"),
        (C_ACTIVE,  "Active"),
        (C_AT_SLOT, "In slot ✓"),
        (C_FAILING, "Failing !"),
        (C_FAILED,  "Failed"),
        (C_COMM,    "Comm link"),
        (C_FORM_LINE,"Formation line"),
    ]
    y = 48
    for col, txt in items:
        pygame.draw.circle(screen, col, (WIDTH-120, y), 5)
        screen.blit(font_small.render(txt, True, col), (WIDTH-110, y-7))
        y += 20
