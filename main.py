import pygame as pg
import matplotlib.pyplot as plt
from config import surface, font, clock, FPS
from physics import space, draw_options
from game import agent, Game


game = Game()
generation = 0
best_score = 0
episode_rewards = []

while True:
    surface.fill(pg.Color('white'))
    for event in pg.event.get():
        if event.type == pg.QUIT:
            agent.save()
            pg.quit()
            plt.plot(episode_rewards)
            plt.xlabel("Generation")
            plt.ylabel('Reward')
            plt.title('Training')
            plt.grid()
            plt.show()
            exit()

    game.update()
    space.step(1 / FPS)
    space.debug_draw(draw_options(surface))

    surface.blit(font.render(f"Gen: {generation}", True, (255, 0, 255)), (10, 10))
    surface.blit(font.render(f"Blocks: {game.placed_blocks}", True, (255, 0, 255)), (10, 40))
    surface.blit(font.render(f"Best: {best_score}", True, (255, 0, 255)), (10, 70))
    surface.blit(font.render(f"Epsilon: {agent.epsilon:.2f}", True, (255, 0, 255)), (10, 100))

    pg.display.flip()
    clock.tick(4000)

    if game.finished:
        r = game.get_reward()
        print(r)
        episode_rewards.append(r)
        next_state = agent.get_state(game.blocks)
        agent.learn(game.prev_state, game.prev_action, r, next_state)
        best_score = max(best_score, game.placed_blocks)
        generation += 1
        agent.decay_epsilon()
        pg.time.delay(300)
        game.reset()