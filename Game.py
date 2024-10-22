# Ping-Pong game with pygame module.
# I actually copy-paste it XD

import random
import pygame
import numpy as np
# Score varibales

player_a_score = 0
player_b_score = 0

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

PADDLE_BUFFER = 10

#speeds of our paddle and ball
PADDLE_SPEED = 8 #4
BALL_X_SPEED = 3
BALL_Y_SPEED = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
IDK_BAll_COLOR = (255, 145, 134)


BALL_HEIGHT = 10
BALL_WIDTH = 10

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


def draw_left_puddle(rightPaddleYPos) :
   #create it
    left_daddle = pygame.Rect(PADDLE_BUFFER, rightPaddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    #draw it
    pygame.draw.rect(screen, WHITE, left_daddle)

# Creating a right paddle for the game
def draw_right_puddle(leftPaddleYPos) :
    right_paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, leftPaddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    #draw it
    pygame.draw.rect(screen, WHITE, right_paddle)

# Creating a pong ball for the game
def draw_ball(ballXPos, ballYPos):
    #small rectangle, create it
    ball = pygame.Rect(ballXPos, ballYPos, 10, 10)
    #draw it
    pygame.draw.rect(screen, WHITE, ball)

# Creating a pen for updating the Score

def update_ball(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection):

    #update the x and y position
    ballXPos = ballXPos + ballXDirection * BALL_X_SPEED
    ballYPos = ballYPos + ballYDirection * BALL_Y_SPEED
    score = 0

    #checks for a collision, if the ball hits the left side, our learning agent
    if (
                        ballXPos <= PADDLE_BUFFER + PADDLE_WIDTH and ballYPos + BALL_HEIGHT >= paddle1YPos and ballYPos - BALL_HEIGHT <= paddle1YPos + PADDLE_HEIGHT):
        #switches directions
        ballXDirection = 1
    #past it
    elif (ballXPos <= 0):
        #negative score
        ballXDirection = 1
        score = -1
        return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]
    
    #check if hits the other side
    if (
                        ballXPos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and ballYPos + BALL_HEIGHT >= paddle2YPos and ballYPos - BALL_HEIGHT <= paddle2YPos + PADDLE_HEIGHT):
        #switch directions
        ballXDirection = -1
    #past it
    elif (ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
        #positive score
        ballXDirection = -1
        score = 1
        return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]
    
    #if it hits the top
    #move down
    if (ballYPos <= 0):
        ballYPos = 0;
        ballYDirection = 1;
    #if it hits the bottom, move up
    elif (ballYPos >= WINDOW_HEIGHT - BALL_HEIGHT):
        ballYPos = WINDOW_HEIGHT - BALL_HEIGHT
        ballYDirection = -1
    return [score, paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection]

#update the paddle position
def update_left_paddle(action, paddle1YPos):
    print(action)
    #if move up
    if (action[1] == float(1)):
        paddle1YPos = paddle1YPos - PADDLE_SPEED
    #if move down
    if (action[2] == float(1)):
        paddle1YPos = paddle1YPos + PADDLE_SPEED

    #don't let it move off the screen
    if (paddle1YPos < 0):
        paddle1YPos = 0
    if (paddle1YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle1YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return paddle1YPos


def update_right_paddle(paddle2YPos, ballYPos):
    #move down if ball is in upper half
    if (paddle2YPos + PADDLE_HEIGHT/2 < ballYPos + BALL_HEIGHT/2):
        paddle2YPos = paddle2YPos + PADDLE_SPEED
    #move up if ball is in lower half
    if (paddle2YPos + PADDLE_HEIGHT/2 > ballYPos + BALL_HEIGHT/2):
        paddle2YPos = paddle2YPos - PADDLE_SPEED
    #don't let it hit top
    if (paddle2YPos < 0):
        paddle2YPos = 0
    #dont let it hit bottom
    if (paddle2YPos > WINDOW_HEIGHT - PADDLE_HEIGHT):
        paddle2YPos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return paddle2YPos


#game class
class PongGame:
    def __init__(self):
        #random number for initial direction of ball
        num = random.randint(0,9)
        #keep score
        self.tally = 0
        #initialie positions of paddle
        self.paddle1YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2YPos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        #and ball direction
        self.ballXDirection = 1
        self.ballYDirection = 1
        #starting point
        self.ballXPos = WINDOW_WIDTH/2 - BALL_WIDTH/2

        #randomly decide where the ball will move
        if(0 < num < 3):
            self.ballXDirection = 1
            self.ballYDirection = 1
        if (3 <= num < 5):
            self.ballXDirection = -1
            self.ballYDirection = 1
        if (5 <= num < 8):
            self.ballXDirection = 1
            self.ballYDirection = -1
        if (8 <= num < 10):
            self.ballXDirection = -1
            self.ballYDirection = -1
        #new random number
        num = random.randint(0,9)
        #where it will start, y part
        self.ballYPos = num*(WINDOW_HEIGHT - BALL_HEIGHT)/9

    #
    def getPresentState(self):
        #for each frame, calls the event queue, like if the main window needs to be repainted
        pygame.event.pump()
        #make the background black
        screen.fill(BLACK)
        #draw our paddles
        draw_left_puddle(self.paddle1YPos)
        draw_right_puddle(self.paddle2YPos)
        #draw our ball
        draw_ball(self.ballXPos, self.ballYPos)
        #copies the pixels from our surface to a 3D array. we'll use this for RL
        state = np.array([self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballYDirection, self.ballXDirection])
        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        #updates the window

        pygame.display.flip()
        #return our surface data
        return state

    #update our screen
    def getNextState(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        #update our paddle
        self.paddle1YPos = update_left_paddle(action, self.paddle1YPos)
        draw_left_puddle(self.paddle1YPos)
        #update evil AI paddle
        self.paddle2YPos = update_right_paddle(self.paddle2YPos, self.ballYPos)
        draw_right_puddle(self.paddle2YPos)
        #update our vars by updating ball position
        [score, self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = update_ball(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection)
        #draw the ball
        draw_ball(self.ballXPos, self.ballYPos)
        #get the surface data
        new_state = np.array([self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballYDirection, self.ballXDirection])
        # image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        #update the window
        pygame.display.flip()
        #record the total score
        self.tally = self.tally + score
        # print ("Tally is" + str(self.tally))
        #return the score and the surface data
        return [score, new_state]
