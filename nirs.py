import pygame as pg
import pymunk.pygame_util
import pymunk
import numpy as np
from random import randint

# Настройки
RES = WIDTH, HEIGHT = 900, 720
FPS = 150
BLOCK_SIZE = 60
START_X = WIDTH // 2
START_Y = 350
MAX_BLOCKS = 50

pg.init()
surface = pg.display.set_mode(RES)
pg.display.set_caption("Q-обучение пирамида")
clock = pg.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(surface)
font = pg.font.Font(None, 28)

pymunk.pygame_util.positive_y_is_up = False

# Физика
space = pymunk.Space()
space.gravity = 0, 8000
platform = pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 26)
platform.elasticity = 0.0
platform.friction = 1.0
space.add(platform)


def create_block(x, y):
    mass = 1
    size = (BLOCK_SIZE, BLOCK_SIZE)
    moment = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, moment)
    body.position = x, y
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = 1.0
    shape.elasticity = 0.0
    shape.color = (244, 193, 193, 255)
    space.add(body, shape)
    return body

# Q-learning агент
class QAgent:
    def __init__(self):
        pass

    def get_state(self, blocks):
        pass

    def choose_action(self, state):
            pass

    def ensure_state_exists(self, state):
        pass

    def learn(self, prev_state, action, reward, next_state):
        pass
    def decay_epsilon(self, factor=0.999, min_eps=0.05):
        pass

agent = QAgent()

# Игровой процесс
# В классе Game:
class Game:
    def __init__(self):
        pass

    def reset(self):
        pass
    def drop_block(self):
        pass
    def update(self):
        pass
    def is_invalid(self, block):
        pass
    def get_reward(self):
        pass

game = Game()
generation = 0
best_score = 0

# Основной цикл
while True:
    surface.fill(pg.Color('white'))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()

    game.update()
    space.step(1 / FPS)
    space.debug_draw(draw_options)

    surface.blit(font.render(f"Gen: {generation}", True, (255, 0, 255)), (10, 10))
    surface.blit(font.render(f"Blocks: {game.placed_blocks}", True, (255, 0, 255)), (10, 40))
    surface.blit(font.render(f"Best: {best_score}", True, (255, 0, 255)), (10, 70))
    surface.blit(font.render(f"Epsilon: {agent.epsilon:.2f}", True, (255, 0, 255)), (10, 100))

    pg.display.flip()
    clock.tick(FPS)

    if game.finished:
        r = game.get_reward()
        next_state = agent.get_state(game.blocks)
        agent.learn(game.prev_state, game.prev_action, r, next_state)

        best_score = max(best_score, game.placed_blocks)
        generation += 1
        agent.decay_epsilon()
        pg.time.delay(300)
        game.reset()