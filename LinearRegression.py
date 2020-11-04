from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import pygame
import pygame.gfxdraw


pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Linear Regression")
running = True
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
classifier = LinearRegression()


def SKlearnGradientDescent(screen, dataX, dataY):
    x = np.array(dataX).reshape(-1, 1)
    y = np.array(dataY)
    classifier.fit(x, y)
    endY = 500 * classifier.coef_[0] + classifier.intercept_

    pygame.draw.aaline(
        screen, WHITE, (0, classifier.intercept_), (500, endY), 2)


def draw(screen, dataX, dataY):
    screen.fill(BLACK)

    if len(dataX) > 0:
        SKlearnGradientDescent(screen, dataX, dataY)
    for x, y in zip(dataX, dataY):
        pygame.draw.circle(screen,  WHITE, (x, y), 3)

    pygame.display.update()


dataX = []
dataY = []
while running:
    draw(screen, dataX, dataY)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            clickPos = pygame.mouse.get_pos()
            dataX.append(clickPos[0])
            dataY.append(clickPos[1])


pygame.quit()
