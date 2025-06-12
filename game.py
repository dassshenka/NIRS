from utils import create_block
from physics import space
from config import HEIGHT, BLOCK_SIZE, START_Y, MAX_BLOCKS, FPS
from agent import QAgent

agent = QAgent()
agent.load()

class Game:
    def __init__(self):
        self.blocks = []
        self.timer = 0
        self.interval = 60
        self.placed_blocks = 0
        self.finished = False
        # TODO увеличим state
        self.prev_state = agent.get_state(self.blocks)
        self.prev_action = None

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
        if not self.blocks:
            return False
        xs = [b.position.x for b in self.blocks]
        min_x, max_x = min(xs), max(xs)
        x = block.position.x
        y = block.position.y
        if x - max_x >= 3 * BLOCK_SIZE or min_x - x >= 3 * BLOCK_SIZE:
            return True
        return y > HEIGHT

    # TODO усложним награду
    def get_reward(self):
        if not self.blocks or self.is_invalid(self.blocks[-1]):
            return -10

        ys = [b.position.y for b in self.blocks]
        xs = [b.position.x for b in self.blocks]
        cx = sum(xs) / len(xs)
        left_x = min(xs)
        right_x = max(xs)
        height = (HEIGHT - min(ys)) // 10
        reward = height // 5
        if right_x == left_x:
            pyramid_score = 0
        else:
            pyramid_score = ((cx - left_x) / ((right_x - left_x) / 2))
        #print(height // 5, pyramid_score * 25)
        reward += pyramid_score * len(self.blocks)
        #h = agent.get_state(self.blocks)
        #max_ind, max_val = max(enumerate(h), key=lambda x: x[1])
        #i = max_ind
        #while i > 0:
        #    if h[i-1] <= h[i]:
        #        reward += h[i]
        #    else:
        #        reward -= h[i]
        #    i -= 1
        #i = max_ind
        #while i > len(h):
        #    if h[i+1] <= h[i]:
        #        reward += h[i]
        #    else:
        #        reward -= h[i]
        #    i += 1

        #print(reward)
        return reward
