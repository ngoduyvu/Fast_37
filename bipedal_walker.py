import os
import numpy as np
import gym
from gym import wrappers
from model import Model
from evostra import EvolutionStrategy

class Agent:
    AGENT_HISTORY_LENGTH = 1
    POPULATION_SIZE = 20
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.01
    INITIAL_EXPLORATION = 1.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 1000000
	ENV_NAME = 'BipedalWalker-v2'
	
	# Initialise the model
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.model = Model()
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION
	
	# Predict new action
    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))
        return prediction

	# Run the agent on the environment and return the rewards
    def play(self, episodes, render=True):
        self.model.set_weights(self.es.weights)
        for episode in xrange(episodes):
            total_reward = 0
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)
            print("total reward:", total_reward)
	
	# Run the training model
    def train(self, iterations):
        self.es.run(iterations, print_step=1)		
		
	# Function calculate the rewards based on the total and number of espisols  
    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in xrange(self.EPS_AVG):
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward/self.EPS_AVG	
