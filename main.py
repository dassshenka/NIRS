# Раздел: Основной игровой цикл с визуализацией и обучением агента
# Назначение: Запуск игрового процесса, отображение состояния игры и графика наград с использованием Pygame и Matplotlib
# Входные данные:
#   surface, font, clock, FPS, SHOW_EVERY - параметры из config
#   space, draw_options - физическое пространство и настройки отрисовки из physics
#   agent, Game - агент Q-обучения и класс игры
# Выходные данные:
#   Отрисовка игрового окна и графика наград, сохранение состояния агента при выходе

import pygame as pg
import matplotlib.pyplot as plt
from config import surface, font, clock, FPS, SHOW_EVERY
from physics import space, draw_options
from game import agent, Game

# Инициализация игры и переменных для статистики
game = Game()
generation = 0
best_score = 0
episode_rewards = []

while True:
    # Очистка экрана: белый фон для визуализации каждые SHOW_EVERY поколений, иначе чёрный
    if generation % SHOW_EVERY == 0:
        surface.fill(pg.Color('white'))
    else:
        surface.fill(pg.Color('black'))

    # Обработка событий Pygame (например, закрытие окна)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            # Сохранение состояния агента перед выходом
            agent.save()
            pg.quit()

            # Построение графика наград по поколениям с помощью Matplotlib
            plt.plot(episode_rewards)
            plt.xlabel("Generation")
            plt.ylabel('Reward')
            plt.title('Training')
            plt.grid()
            plt.show()
            exit()

    # Обновление состояния игры и физики
    game.update()
    space.step(1 / FPS)
    # Отрисовка физического пространства и информации на экране каждые SHOW_EVERY поколений
    if generation % SHOW_EVERY == 0:
        space.debug_draw(draw_options(surface))

        surface.blit(font.render(f"Gen: {generation}", True, (255, 0, 255)), (10, 10))
        surface.blit(font.render(f"Blocks: {game.placed_blocks}", True, (255, 0, 255)), (10, 40))
        surface.blit(font.render(f"Best: {best_score}", True, (255, 0, 255)), (10, 70))
        surface.blit(font.render(f"Epsilon: {agent.epsilon:.2f}", True, (255, 0, 255)), (10, 100))

        pg.display.flip()

    # Контроль частоты кадров
    clock.tick(10000)
    # Если игра завершена, обработка результатов эпизода
    if game.finished:
        r = game.get_reward()
        episode_rewards.append(r)
        next_state = agent.get_state(game.blocks)
        agent.learn(game.prev_state, game.prev_action, r, next_state)
        best_score = max(best_score, game.placed_blocks)
        generation += 1

        # Понижение epsilon для уменьшения случайных действий с течением времени
        agent.decay_epsilon()
        #pg.time.delay(300)
        # Сброс состояния игры для следующего эпизода
        game.reset()