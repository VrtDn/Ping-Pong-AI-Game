import torch
import random
import numpy as np
from collections import deque 
from Game import PongGame
from Model import Linear_QNet, Qtrainer
from PlotHelper import plot 

#hyper params
ACTIONS = 3 #up,down, stay
#define our learning rate
GAMMA = 0.99
#for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
#how many frames to anneal epsilon
EXPLORE = 500000 
OBSERVE = 50000 #50000
#store our experiences, the size of it
REPLAY_MEMORY = 500000
#batch size to train on
BATCH = 100
LR = 0.0001

class Agent :
    def __init__(self) -> None:
        self.num_of_games = 0
        self.epsilon = INITIAL_EPSILON #randomess
        self.gamma = GAMMA #discount_rate
        self.memory = deque(maxlen=REPLAY_MEMORY)
        self.record = 0
        self.t = 0
        self.model = Linear_QNet(hidden_size1=128, hidden_size2 = 64, hidden_size3=32, input_size=6, output_size=ACTIONS)
        self.trainer = Qtrainer(model=self.model, lr=LR, gamma=self.gamma)
        self.remember = []

    def train_memory(self, reward, state, next_state, action) :
        self.memory.append((state, action, reward, next_state))
        if self.t > OBSERVE :
            mini_sample = random.sample(self.memory, BATCH) # list of tuples
            states, actions, rewards, next_states = zip(*mini_sample)
            self.trainer.train_steps(states, actions, rewards, next_states)
        self.t = self.t+1


    def get_action(self, state) :
        # random moves between exploration and exploitation
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state)
        if(random.random() <= self.epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(prediction.detach().numpy())
        prediction[maxIndex] = 1
        
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        return prediction


def main_loop_train():
    print("train starts")
    agent = Agent()
    game = PongGame()

    while True :
        state = game.getPresentState()
        action = agent.get_action(state)

        reward, next_state = game.getNextState(action)

        agent.train_memory(reward = reward, action=action, next_state=next_state,state=state)
        

        if reward < 0 :
            agent.num_of_games += 1
            print("game number "+ str(agent.num_of_games))




if __name__ == "__main__":
    main_loop_train()