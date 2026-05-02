import numpy as np
import pygame

WIDTH, HEIGHT = 800, 600

class Drone:
    def __init__(self, x, y, idx):
        self.position = np.array([x, y], dtype=float)
        self.velocity = (np.random.rand(2) - 0.5) * 4

        self.max_speed = 4
        self.perception_radius = 60
        self.separation_radius = 25

        self.active = True
        self.idx = idx

    def update(self, drones, target, total_drones, mode):
        if not self.active:
            return

        separation = np.zeros(2)
        alignment = np.zeros(2)
        cohesion = np.zeros(2)

        total_neighbors = 0

        for other in drones:
            if other is self or not other.active:
                continue

            distance = np.linalg.norm(self.position - other.position)

            if distance < self.perception_radius:
                alignment += other.velocity
                cohesion += other.position
                total_neighbors += 1

                if distance < self.separation_radius:
                    separation += (self.position - other.position) / (distance + 1e-5)

        if total_neighbors > 0:
            alignment /= total_neighbors
            alignment -= self.velocity

            cohesion /= total_neighbors
            cohesion -= self.position

        # ---------------- MODES ---------------- #

        if mode == "swarm":
            target_force = target - self.position
            self.velocity += target_force * 0.005

        elif mode == "circle":
            angle = (2 * np.pi / total_drones) * self.idx
            radius = 120  # bigger + clearer

            pattern_point = np.array([
                target[0] + radius * np.cos(angle),
                target[1] + radius * np.sin(angle)
            ])

            pattern_force = pattern_point - self.position
            self.velocity += pattern_force * 0.02  # stronger pull

        elif mode == "leader":
            leader = drones[0]

            if self.idx == 0:
                # Leader moves to target
                target_force = target - self.position
                self.velocity += target_force * 0.01
            else:
                # Followers follow leader
                follow_force = leader.position - self.position
                self.velocity += follow_force * 0.008

        # ---------------- COMMON FORCES ---------------- #

        self.velocity += separation * 0.6
        self.velocity += alignment * 0.05
        self.velocity += cohesion * 0.01

        # small randomness
        self.velocity += (np.random.rand(2) - 0.5) * 0.03

        # limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

        # update position
        self.position += self.velocity

        # boundary
        if self.position[0] < 0 or self.position[0] > WIDTH:
            self.velocity[0] *= -1
        if self.position[1] < 0 or self.position[1] > HEIGHT:
            self.velocity[1] *= -1

        self.position[0] = np.clip(self.position[0], 0, WIDTH)
        self.position[1] = np.clip(self.position[1], 0, HEIGHT)

    def draw(self, screen):
        if not self.active:
            color = (255, 0, 0)
        elif self.idx == 0:
            color = (255, 255, 0)  # leader = yellow ⭐
        else:
            speed = np.linalg.norm(self.velocity)
            intensity = min(255, int(speed * 60))
            color = (0, intensity, 255)

        pygame.draw.circle(screen, color, self.position.astype(int), 6)

        # direction line
        end_pos = self.position + self.velocity * 5
        pygame.draw.line(screen, (255, 255, 255),
                         self.position.astype(int),
                         end_pos.astype(int), 2)