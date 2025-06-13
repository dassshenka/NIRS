# Раздел: Инициализация физического пространства и создание платформы
# Назначение: Создание пространства с гравитацией и добавление статической платформы
# Выходные данные:
#   space (pymunk.Space) - объект физического пространства с добавленной платформой
#   draw_options (pymunk.pygame_util.DrawOptions) - настройки отрисовки для pygame

import pymunk
import pymunk.pygame_util
from config import WIDTH, HEIGHT

# Создание физического пространства
space = pymunk.Space()

# Установка гравитации (ось Y направлена вниз, поэтому положительное значение гравитации вниз)
space.gravity = 0, 8000

# Настройка отрисовки: положительная ось Y направлена вниз (соответствует экранным координатам)
pymunk.pygame_util.positive_y_is_up = False

# Создание объекта для отрисовки физических объектов в pygame
draw_options = pymunk.pygame_util.DrawOptions

# Создание статической платформы (сегмента) внизу экрана
platform = pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1)

# Установка физических свойств платформы
platform.elasticity = 0.0  # Отсутствие упругости (не отскакивает)
platform.friction = 1.0    # Коэффициент трения

# Добавление платформы в физическое пространство
space.add(platform)