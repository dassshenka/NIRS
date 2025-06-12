import pymunk.pygame_util
import pymunk
from config import WIDTH, HEIGHT

space = pymunk.Space()
space.gravity = 0, 8000
pymunk.pygame_util.positive_y_is_up = False
draw_options = pymunk.pygame_util.DrawOptions

platform = pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1)
platform.elasticity = 0.0
platform.friction = 1.0
space.add(platform)
