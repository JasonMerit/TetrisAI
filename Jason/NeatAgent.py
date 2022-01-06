# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:04:52 2022

@author: Jason
"""
import pygame
import random
import os
from Tetris import Tetris
import numpy as np
import time
import neat
import pickle


gen = 0

def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    agents and sets their fitness based on score.
    """
    global gen
    gen += 1

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # environment that uses that network to play
    nets = np.array([])
    envs = np.array([])
    ge = np.array([])
    
    for _, genome in genomes:            
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets = np.append(nets, net)
        envs = np.append(envs, Tetris(False))
        ge = np.append(ge, genome)
        
    rem = np.empty(len(envs))
    best_agent = None
    while len(envs) > 0:
        # Have each env take a step
        for x, env in enumerate(envs):
            input = tuple(env.get_state())
            output = nets[x].activate(input) # Returns a tuple of best action estimation
            action = output.index(max(output)) # Take max estimated action
            reward, done = env.step(action)
            env.drop()
            
            # Update relevant sizes
            ge[x].fitness += reward
            rem[x] = done
        
        # Remove loser envs
        best_agent = nets[0]
        nets = nets[rem == 0]
        envs = envs[rem == 0]
        ge = ge[rem == 0]
        
        num_agents_left = len(envs)
        rem = np.empty(num_agents_left)
        
    pickle.dump(best_agent,open("best.pickle", "wb"))

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play Tetris.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)