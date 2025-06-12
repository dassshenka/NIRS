import pygame as pg

# Настройки
RES = WIDTH, HEIGHT = 900, 900
FPS = 150
#TODO уменьшим block size
BLOCK_SIZE = 125
START_X = WIDTH // 2
START_Y = 30
#TODO увеличим max blocks
MAX_BLOCKS = 15
COLS = WIDTH // BLOCK_SIZE

# Pygame
pg.init()
surface = pg.display.set_mode(RES)
pg.display.set_caption("Q-обучение пирамида")
clock = pg.time.Clock()
font = pg.font.Font(None, 28)