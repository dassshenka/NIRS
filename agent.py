import numpy as np
import pickle
import os
from config import BLOCK_SIZE, WIDTH, HEIGHT, COLS
from random import randint


class QAgent:
    def __init__(self):
        #TODO увеличим actions
        self.actions = [i for i in range(BLOCK_SIZE // 2, 900, BLOCK_SIZE)]  # дискретные X
        self.q_table = {}
        self.epsilon = 1.0
        self.alpha = 0.1        # скорость обучения
        self.gamma = 0.95       # важность будущих наград

    def get_state(self, blocks):
        if not blocks:
            return tuple([0] * (COLS + 1))
        # TODO увеличим state
        heights = [0] * COLS
        for block in blocks:
            x_idx = int(block.position.x // BLOCK_SIZE)
            if x_idx < 0 or x_idx > WIDTH // BLOCK_SIZE - 1:  # если блок упал
                heights[-1] = 1
                continue
            y_pos = block.position.y
            h = HEIGHT - y_pos
            heights[x_idx] = max(heights[x_idx], int(h // BLOCK_SIZE) + 1 )
        #print(heights)
        return tuple(heights)

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
        #print(prev_state, next_state)
        self.ensure_state_exists(prev_state)
        self.ensure_state_exists(next_state)
        predict = self.q_table[prev_state][action]  # оцениваем ошибку предсказания
        target = reward + self.gamma * np.max(self.q_table[next_state])  # оцениваем правду (награда сейчаас + лучшая ожидаемая награда из нового состояния)
        self.q_table[prev_state][action] += self.alpha * (target - predict)

    # TODO увеличим factor
    def decay_epsilon(self, factor=0.999, min_eps=0.1):
        self.epsilon = max(min_eps, self.epsilon * factor)

    def save(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename='q_table.pkl'):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
