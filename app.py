from turtle import Screen, isdown, xcor
from urllib.request import ProxyHandler
from matplotlib import testing
import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
from yaml import load


WINDOWSIZEX = 640
WINDOWSIZEY = 480


BOUNDRYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

IMAGESAVE = False

MODEL = load_model("handwritten.h5")

LABLES = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

#Initialize pygame
pygame.init()

FONT = pygame.font.Font(None,18)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit board")

iswriting = False

num_xcord = []
num_ycord = []

img_count = 1

PREDICT = True

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord,ycord),4,0)

            num_xcord.append(xcord)
            num_ycord.append(ycord)

        if event.type  == MOUSEBUTTONDOWN:
            iswriting = True

        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            num_xcord = sorted(num_xcord)
            num_ycord = sorted(num_ycord)

            rect_min_x , rect_max_x = max(num_xcord[0]-BOUNDRYINC, 0), min(WINDOWSIZEX, num_xcord[-1]+BOUNDRYINC)
            rect_min_y , rect_max_y = max(num_ycord[0]-BOUNDRYINC, 0), min(WINDOWSIZEY, num_ycord[-1]+BOUNDRYINC)


            num_xcord = []
            num_ycord = []

            img_arr =  np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                img_count += 1

            if PREDICT:

                image = cv2.resize(img_arr,(28,28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28,28))/255

                label = str(LABLES[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurf = FONT.render(label, True, RED, WHITE)
                textRecObj = pygame.Surface.get_rect(textSurf)
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                DISPLAYSURF.blit(textSurf, textRecObj)

            if event.type == KEYDOWN:
                if event.unicode == 'n':
                    DISPLAYSURF.fill(BLACK)

    pygame.display.update()