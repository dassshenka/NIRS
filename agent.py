# Раздел: Класс QAgent для Q-обучения
# Назначение: Реализация агента Q-обучения для управления блоками в дискретном пространстве
# Входные данные:
#   blocks (list) - список объектов блоков с атрибутом position (с координатами x, y)
# Выходные данные:
#   Методы класса обеспечивают выбор действия, обновление Q-таблицы, сохранение и загрузку состояния агента

import numpy as np
import pickle
import os
from config import BLOCK_SIZE, WIDTH, HEIGHT, COLS
from random import randint


class QAgent:
    def __init__(self):
        """
            Инициализация агента:
            - actions: список дискретных действий по оси X (TODO: увеличить количество действий)
            - q_table: словарь Q-таблицы (состояния -> значения действий)
             - epsilon: коэффициент исследования (жадность)
            - alpha: скорость обучения
            - gamma: коэффициент дисконтирования будущих наград
        """
        self.actions = [i for i in range(BLOCK_SIZE // 2, 900, BLOCK_SIZE)]  # дискретные X
        self.q_table = {}
        self.epsilon = 1.0
        self.alpha = 0.2        # скорость обучения
        self.gamma = 0.95       # важность будущих наград

    def get_state(self, blocks):
        """
        Формирование состояния на основе текущих блоков.
        Если блоков нет, возвращает состояние с нулями.
        TODO: увеличить размерность состояния.

        Аргументы:
            blocks (list): список блоков с координатами position.x и position.y

        Возвращает:
            tuple: кортеж высот по колонкам
        """
        if not blocks:
            return tuple([0] * (COLS + 1))

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
        """
        Выбор действия на основе ε-жадной стратегии.
        С вероятностью epsilon выбирается случайное действие,
        иначе выбирается действие с максимальным значением Q.

        Аргументы:
        state (tuple): текущее состояние

        Возвращает:
        int: индекс выбранного действия
        """
        if randint(0, 100) < int(self.epsilon * 100):
            return randint(0, len(self.actions) - 1)
        else:
            self.ensure_state_exists(state)
            return int(np.argmax(self.q_table[state]))

    def ensure_state_exists(self, state):
        """
        Проверка наличия состояния в Q-таблице.
        Если состояние отсутствует, инициализирует нулевой вектор значений действий.

        Аргументы:
            state (tuple): состояние для проверки
        """
        if state not in self.q_table:
            #print('Неизвестное состояние')
            self.q_table[state] = np.zeros(len(self.actions))


    def learn(self, prev_state, action, reward, next_state):
        """
        Обновление Q-значений по формуле Q-обучения.

        Аргументы:
            prev_state (tuple): предыдущее состояние
            action (int): выполненное действие
            reward (float): полученная награда
            next_state (tuple): новое состояние
        """
        #print(prev_state, next_state)
        self.ensure_state_exists(prev_state)
        self.ensure_state_exists(next_state)
        predict = self.q_table[prev_state][action]  # оцениваем ошибку предсказания
        target = reward + self.gamma * np.max(self.q_table[next_state])  # оцениваем правду (награда сейчаас + лучшая ожидаемая награда из нового состояния)
        self.q_table[prev_state][action] += self.alpha * (target - predict)


    def decay_epsilon(self, factor=0.9999, min_eps=0.1):
        """
        Понижение коэффициента исследования epsilon с заданным фактором и минимальным значением.

        Аргументы:
            factor (float): множитель для уменьшения epsilon (TODO: увеличить)
            min_eps (float): минимальное значение epsilon
        """
        self.epsilon = max(min_eps, self.epsilon * factor)

    def save(self, filename='q_table.pkl'):
        """
        Сохранение Q-таблицы в файл.
        Аргументы:
            filename (str): имя файла для сохранения
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, filename='q_table.pkl'):
        """
        Загрузка Q-таблицы из файла, если файл существует.

        Аргументы:
            filename (str): имя файла для загрузки
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
