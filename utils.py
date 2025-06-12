import pymunk
from config import BLOCK_SIZE, START_Y
from physics import space

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
