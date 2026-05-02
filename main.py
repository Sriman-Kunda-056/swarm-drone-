import pygame
import random
from drone import Drone

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Swarm Drone Simulation")

clock = pygame.time.Clock()

NUM_DRONES = 30
drones = [Drone(WIDTH // 2, HEIGHT // 2, i) for i in range(NUM_DRONES)]

mode = "swarm"

font = pygame.font.SysFont(None, 30)

running = True
while running:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                mode = "swarm"
            if event.key == pygame.K_2:
                mode = "circle"
            if event.key == pygame.K_3:
                mode = "leader"
            if event.key == pygame.K_r:
                drones = [Drone(WIDTH // 2, HEIGHT // 2, i) for i in range(NUM_DRONES)]
            if event.key == pygame.K_f:
                random.choice(drones).active = False

    screen.fill((0, 0, 0))

    target = pygame.mouse.get_pos()
    target = pygame.math.Vector2(target)

    # random failure
    if random.random() < 0.005:
        random.choice(drones).active = False

    for drone in drones:
        drone.update(drones, target, NUM_DRONES, mode)
        drone.draw(screen)

    pygame.draw.circle(screen, (255, 0, 0), target, 6)

    text = font.render(f"Mode: {mode}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

pygame.quit()
