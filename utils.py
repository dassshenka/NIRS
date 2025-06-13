# Раздел: Модуль создания блоков физического пространства
# Назначение: Создание физического блока с заданными параметрами массы, размера и свойств
# Входные данные:
#   x (float) - координата X центра блока
#   y (float) - координата Y центра блока
# Выходные данные:
#   body (pymunk.Body) - объект физического тела, добавленного в пространство

import pymunk
from config import BLOCK_SIZE, START_Y
from physics import space


def create_block(x, y):
    """
    Функция создает физический блок с параметрами:
    - масса: 1
    - размер: BLOCK_SIZE x BLOCK_SIZE
    - трение: 1.0
    - упругость: 0.0
    - цвет: светло-розовый (RGBA: 244, 193, 193, 255)

    Параметры:
    x (float): координата X центра блока
    y (float): координата Y центра блока

    Возвращает:
    pymunk.Body: объект физического тела, добавленного в пространство
    """

    mass = 1
    size = (BLOCK_SIZE, BLOCK_SIZE)

    # Вычисление момента инерции для прямоугольника
    moment = pymunk.moment_for_box(mass, size)

    # Создание тела с заданной массой и моментом инерции
    body = pymunk.Body(mass, moment)
    body.position = x, y

    # Создание формы прямоугольника, связанной с телом
    shape = pymunk.Poly.create_box(body, size)
    shape.friction = 1.0  # Коэффициент трения
    shape.elasticity = 0.0  # Коэффициент упругости
    shape.color = (244, 193, 193, 255)  # Цвет блока в формате RGBA

    # Добавление тела и формы в физическое пространство
    space.add(body, shape)

    return body