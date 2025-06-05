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
        self.actions = [i for i in range(200, 700, BLOCK_SIZE)]  # дискретные X
        self.q_table = {}
        self.epsilon = 1.0
        self.alpha = 0.1        # скорость обучения
        self.gamma = 0.95       # важность будущих наград

    def get_state(self, blocks):
        if not blocks:
            return START_X, START_Y

        xs = [b.position.x for b in blocks]
        ys = [b.position.y for b in blocks]
        total_mass = len(blocks)

        cx = int(sum(xs) / total_mass)
        cy = int(sum(ys) / total_mass)

        return cx, cy

    def choose_action(self, state):
        if randint(0, 100) < int(self.epsilon * 100):
            return randint(0, len(self.actions) - 1)
        else:
            self.ensure_state_exists(state)
            return int(np.argmax(self.q_table[state]))

    def ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

    def learn(self, prev_state, action, reward, next_state):
        self.ensure_state_exists(prev_state)
        self.ensure_state_exists(next_state)
        predict = self.q_table[prev_state][action]  # оцениваем ошибку предсказания
        target = reward + self.gamma * np.max(self.q_table[next_state])  # оцениваем правду (награда сейчаас + лучшая ожидаемая награда из нового состояния)
        self.q_table[prev_state][action] += self.alpha * (target - predict)

    def decay_epsilon(self, factor=0.999, min_eps=0.05):
        self.epsilon = max(min_eps, self.epsilon * factor)


agent = QAgent()


# Игровой процесс
class Game:
    def __init__(self):
        self.blocks = []
        self.timer = 0
        self.interval = 50
        self.placed_blocks = 0
        self.finished = False
        self.prev_state = (0, 0)
        self.prev_action = 0

    def reset(self):
        for b in self.blocks:
            space.remove(b, *b.shapes)
        self.blocks.clear()
        self.timer = 0
        self.placed_blocks = 0
        self.finished = False

    def drop_block(self):
        self.prev_state = agent.get_state(self.blocks)
        action_idx = agent.choose_action(self.prev_state)
        x = agent.actions[action_idx]
        y = START_Y
        self.prev_action = action_idx

        block = create_block(x, y)
        self.blocks.append(block)
        self.placed_blocks += 1

    def update(self):
        if self.finished:
            return

        self.timer += 1
        if self.timer >= self.interval:
            self.timer = 0
            if self.placed_blocks < MAX_BLOCKS:
                self.drop_block()
            else:
                self.finished = True

        for b in self.blocks:
            if self.is_invalid(b):
                self.finished = True
                break

    def is_invalid(self, block):
        xs = [b.position.x for b in self.blocks]
        min_x, max_x = min(xs), max(xs)
        x = block.position.x
        y = block.position.y
        if abs(max_x - x) >= 3 * BLOCK_SIZE or abs(min_x - x) >= 3 * BLOCK_SIZE:
            return True
        return y > HEIGHT

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