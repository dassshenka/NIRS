# Раздел: Класс Game для управления игровым процессом
# Назначение: Реализация логики игры с использованием Q-агента для управления блоками
# Входные данные:
#   Используются константы из config (HEIGHT, BLOCK_SIZE, START_Y, MAX_BLOCKS, FPS)
#   Агент QAgent загружается из внешнего модуля agent
# Выходные данные:
#   Методы класса обеспечивают управление блоками, обновление состояния игры, обучение агента и вычисление награды

from utils import create_block
from physics import space
from config import HEIGHT, BLOCK_SIZE, START_Y, MAX_BLOCKS, FPS
from agent import QAgent

# Инициализация агента Q-обучения и загрузка Q-таблицы
agent = QAgent()
agent.load()

class Game:
    def __init__(self):
        """
        Инициализация игрового состояния:
            - blocks: список текущих блоков в игре
            - timer: счётчик кадров для интервала появления блоков
            - interval: интервал между падениями блоков (в кадрах)
            - placed_blocks: количество размещённых блоков
            - finished: флаг окончания игры
            - prev_state: предыдущее состояние агента (TODO: увеличить размерность состояния)
            - prev_action: предыдущее действие агента
        """
        self.blocks = []
        self.timer = 0
        self.interval = 60
        self.placed_blocks = 0
        self.finished = False
        self.prev_state = agent.get_state(self.blocks)
        self.prev_action = None

    def reset(self):
        """
        Сброс игрового состояния:
            - удаление всех блоков из физического пространства
            - очистка списка блоков
            - сброс таймера и счётчика размещённых блоков
            - сброс флага окончания игры
        """
        for b in self.blocks:
            space.remove(b, *b.shapes)
        self.blocks.clear()
        self.timer = 0
        self.placed_blocks = 0
        self.finished = False

    def drop_block(self):
        """
        Метод создания и падения нового блока:
            - получение текущего состояния агента
            - выбор действия (позиции по X) агентом
            - создание блока в выбранной позиции
            - симуляция падения блока (прогон физики на fall_frames кадров)
            - проверка валидности положения блока
            - обновление списка блоков и счётчика размещённых
            - обучение агента на основе полученной награды
        """
        self.prev_state = agent.get_state(self.blocks)
        action_idx = agent.choose_action(self.prev_state)
        x = agent.actions[action_idx]
        y = START_Y
        self.prev_action = action_idx

        block = create_block(x, y)
        fall_frames = 70
        for _ in range(fall_frames):
            space.step(1 / FPS)
        if self.is_invalid(block):
            self.finished = True
            #space.remove(block, *block.shapes)
            #return
        self.blocks.append(block)
        self.placed_blocks += 1
        next_state = agent.get_state(self.blocks)
        reward = self.get_reward()

        agent.learn(self.prev_state, self.prev_action, reward, next_state)

    def update(self):
        """
        Обновление состояния игры:
            - если игра завершена, выход из метода
            - увеличение таймера
            - при достижении интервала создаётся новый блок (если не достигнут максимум)
            - проверка положения блоков на выход за нижнюю границу экрана
        """
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
            if b.position.y > HEIGHT:
                self.finished = True
                break

    def is_invalid(self, block):
        """
        Проверка валидности положения блока:
            - если блоков нет, всегда валидно
            - вычисляется минимальная и максимальная координата X среди блоков
            - если новый блок слишком далеко по X от существующих (более 3 BLOCK_SIZE), позиция считается невалидной
            - если блок опустился ниже нижней границы экрана, позиция невалидна

        Аргументы:
            block (pymunk.Body): проверяемый блок

        Возвращает:
            bool: True, если позиция невалидна, иначе False
        """
        if not self.blocks:
            return False
        xs = [b.position.x for b in self.blocks]
        min_x, max_x = min(xs), max(xs)
        x = block.position.x
        y = block.position.y
        if x - max_x >= 3 * BLOCK_SIZE or min_x - x >= 3 * BLOCK_SIZE:
            return True
        return y > HEIGHT

    def get_reward(self):
        """
        Вычисление награды для агента (TODO: усложнить формулу награды):
            - если блоков нет или последний блок невалиден, возвращается штраф -10
            - иначе награда рассчитывается на основе высот блоков:
                + увеличивается, если соседний столбец ниже текущего
                - уменьшается, если соседний столбец выше или равен текущему

        Возвращает:
            float: вычисленная награда
        """
        if not self.blocks or self.is_invalid(self.blocks[-1]):
            return -15

        reward = 0

        is_symmetry = True
        h = agent.get_state(self.blocks)
        max_ind, max_val = max(enumerate(h), key=lambda x: x[1])
        i = max_ind
        while i > 0:
            if h[i-1] < h[i]:
                reward += h[i]
            else:
                is_symmetry = False
                reward -= h[i]
            i -= 1
        i = max_ind
        while i > len(h):
            if h[i+1] < h[i]:
                reward += h[i]
            else:
                is_symmetry = False
                reward -= h[i]
            i += 1

        if is_symmetry:
            reward *= 10
        #print(reward)
        return reward
