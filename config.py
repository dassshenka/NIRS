# Раздел: Настройки и инициализация Pygame
# Назначение: Определение параметров экрана, блоков и инициализация библиотеки Pygame

import pygame as pg

# --- Настройки экрана и игры ---
RES = WIDTH, HEIGHT = 900, 900      # Разрешение окна (ширина, высота) в пикселях
FPS = 150                          # Частота обновления кадров в секунду

# --- Параметры блоков ---
BLOCK_SIZE = 125                   # Размер одного блока (TODO: уменьшить размер блока)
START_X = WIDTH // 2               # Начальная позиция X (центр экрана)
START_Y = 30                      # Начальная позиция Y (отступ сверху)
MAX_BLOCKS = 30                   # Максимальное количество блоков (TODO: увеличить)
SHOW_EVERY = 100                   # Частота обновления отображения (каждые N кадров)
COLS = WIDTH // BLOCK_SIZE         # Количество колонок по ширине экрана

# --- Инициализация Pygame ---
pg.init()                         # Инициализация всех модулей Pygame

surface = pg.display.set_mode(RES)  # Создание окна с заданным разрешением
pg.display.set_caption("Q-обучение пирамида")  # Заголовок окна

clock = pg.time.Clock()            # Объект для контроля времени и FPS

font = pg.font.Font(None, 28)     # Шрифт для вывода текста (стандартный, размер 28)