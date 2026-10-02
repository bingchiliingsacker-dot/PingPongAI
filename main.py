import sys
from pathlib import Path
from random import choice, randint

import pygame
import torch

from src_py import AI

from time import perf_counter


# =========================
# CONSTANTS (must match environment.rs)
# =========================

WIDTH = 800
HEIGHT = 600

BLUE_X = 60
RED_X = 730
PADDLE_W = 10
PADDLE_H = 100
PADDLE_HALF_H = PADDLE_H // 2

BALL_SIZE = 30
BALL_HALF = BALL_SIZE // 2

LEFT_GOAL_X = 80     # ball centre <= this  -> red scores
RIGHT_GOAL_X = 720   # ball centre >= this  -> blue scores

MAX_STEPS = 500_000  # same step limit as training (draw)

FPS = 240
NUM_RUNS = 100

BALL_IMAGE_PATH = r"C:\Users\Bianca\OneDrive\Pictures\Screenshots\ball.png"

ROOT = Path(__file__).resolve().parent


# =========================
# AI SETUP
# =========================

# A 6->64->64->3 MLP with batch size 1 is faster on the CPU than on the GPU.
device = torch.device("cpu")


def load_model(name):
    net = AI.NeuralNetwork().to(device)
    net.load_state_dict(
        torch.load(ROOT / name, map_location=device, weights_only=True)
    )
    net.eval()
    return net


blue_net = load_model("blue_model.pth")
red_net = load_model("red_model.pth")


@torch.inference_mode()
def choose_action(net, obs):
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return net(x).argmax(dim=1).item()


# =========================
# GAME LOGIC (identical to Env::step in environment.rs)
# =========================

class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ball_cx = WIDTH // 2
        self.ball_cy = randint(50, HEIGHT - 50)
        self.red_cy = HEIGHT // 2
        self.blue_cy = HEIGHT // 2
        self.dx = choice([-1, 1])
        self.dy = choice([-1, 1])
        self.steps = 0

    # ----- observations (same as write_obs) -----
    def red_obs(self):
        return [
            self.ball_cx / WIDTH,
            self.ball_cy / HEIGHT,
            self.red_cy / HEIGHT,
            float(self.dx),
            float(self.dy),
            (self.ball_cy - self.red_cy) / HEIGHT,
        ]

    def blue_obs(self):
        # Blue sees a horizontally mirrored world.
        return [
            (WIDTH - self.ball_cx) / WIDTH,
            self.ball_cy / HEIGHT,
            self.blue_cy / HEIGHT,
            float(-self.dx),
            float(self.dy),
            (self.ball_cy - self.blue_cy) / HEIGHT,
        ]

    def paddle_hit(self, paddle_x, paddle_cy):
        return (
            paddle_x < self.ball_cx + BALL_HALF
            and paddle_x + PADDLE_W > self.ball_cx - BALL_HALF
            and abs(self.ball_cy - paddle_cy) < BALL_HALF + PADDLE_HALF_H
        )

    def step(self, action_red, action_blue):
        """Returns 'red', 'blue', 'draw' or None."""

        # Paddle collisions (only while the ball travels toward the paddle)
        if self.dx == -1 and self.paddle_hit(BLUE_X, self.blue_cy):
            self.dx = 1
        if self.dx == 1 and self.paddle_hit(RED_X, self.red_cy):
            self.dx = -1

        # Wall collisions
        if self.ball_cy >= HEIGHT - BALL_HALF:
            self.ball_cy = HEIGHT - BALL_HALF
            self.dy = -1
        elif self.ball_cy <= BALL_HALF:
            self.ball_cy = BALL_HALF
            self.dy = 1

        # Move ball
        self.ball_cx += self.dx
        self.ball_cy += self.dy

        # Move paddles (0 = up, 1 = down, 2 = stay), centre clamped
        min_c = PADDLE_HALF_H
        max_c = HEIGHT - PADDLE_HALF_H

        if action_red == 0 and self.red_cy > min_c:
            self.red_cy -= 1
        elif action_red == 1 and self.red_cy < max_c:
            self.red_cy += 1

        if action_blue == 0 and self.blue_cy > min_c:
            self.blue_cy -= 1
        elif action_blue == 1 and self.blue_cy < max_c:
            self.blue_cy += 1

        # Score
        if self.ball_cx >= RIGHT_GOAL_X:
            return "blue"
        if self.ball_cx <= LEFT_GOAL_X:
            return "red"

        self.steps += 1
        if self.steps >= MAX_STEPS:
            return "draw"

        return None


# =========================
# PYGAME
# =========================

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong AI")

clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 30)

try:
    ball_image = pygame.image.load(BALL_IMAGE_PATH)
    ball_image = pygame.transform.scale(ball_image, (BALL_SIZE, BALL_SIZE))
except (pygame.error, FileNotFoundError):
    ball_image = None  # fall back to a drawn circle


def draw(game, message=None, message_color=(0, 0, 0)):
    screen.fill((255, 255, 255))

    # Arena: goal lines are where points are actually scored,
    # top/bottom edges are where the ball really bounces.
    pygame.draw.line(screen, (0, 0, 0), (LEFT_GOAL_X, 0), (LEFT_GOAL_X, HEIGHT), 3)
    pygame.draw.line(screen, (0, 0, 0), (RIGHT_GOAL_X, 0), (RIGHT_GOAL_X, HEIGHT), 3)
    pygame.draw.line(screen, (0, 0, 0), (LEFT_GOAL_X, 2), (RIGHT_GOAL_X, 2), 5)
    pygame.draw.line(screen, (0, 0, 0), (LEFT_GOAL_X, HEIGHT - 3), (RIGHT_GOAL_X, HEIGHT - 3), 5)

    blue_rect = pygame.Rect(BLUE_X, game.blue_cy - PADDLE_HALF_H, PADDLE_W, PADDLE_H)
    red_rect = pygame.Rect(RED_X, game.red_cy - PADDLE_HALF_H, PADDLE_W, PADDLE_H)
    pygame.draw.rect(screen, (0, 0, 255), blue_rect)
    pygame.draw.rect(screen, (255, 0, 0), red_rect)

    top_left = (game.ball_cx - BALL_HALF, game.ball_cy - BALL_HALF)
    if ball_image is not None:
        screen.blit(ball_image, top_left)
    else:
        pygame.draw.circle(screen, (0, 0, 0), (game.ball_cx, game.ball_cy), BALL_HALF)

    if message:
        surf = font.render(message, True, message_color)
        screen.blit(surf, surf.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    pygame.display.flip()


# =========================
# MAIN LOOP
# =========================

blue_wins = 0
red_wins = 0
draws = 0

game = Game()

for run in range(NUM_RUNS):
    game.reset()
    winner = None

    start = perf_counter()
    while winner is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        red_action = choose_action(red_net, game.red_obs())
        blue_action = choose_action(blue_net, game.blue_obs())

        winner = game.step(red_action, blue_action)

        draw(game)
        clock.tick(FPS)

    if winner == "blue":
        blue_wins += 1
        draw(game, "BLUE WON!", (0, 0, 255))
    elif winner == "red":
        red_wins += 1
        draw(game, "RED WON!", (255, 0, 0))
    else:
        draws += 1
        draw(game, "DRAW (step limit)")

    end = perf_counter()

    print(f"--- {winner.capitalize()} {'Won' if winner != 'draw' else ''} ---")
    print("Blue:", (BLUE_X + PADDLE_W // 2, game.blue_cy))
    print("Red:", (RED_X + PADDLE_W // 2, game.red_cy))
    print("Ball:", (game.ball_cx, game.ball_cy))
    print(f'Game lasted {end-start} seconds')
    print()

    pygame.time.wait(2000)


# =========================
# FINAL SCORE
# =========================

print(f"blue_wins: {blue_wins}, red_wins: {red_wins}, draws: {draws}")

pygame.quit()
sys.exit()
