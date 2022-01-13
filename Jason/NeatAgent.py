# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:04:52 2022

@author: Jason
"""


import os
from Tetris import Tetris
import numpy as np

import neat
import pickle
import time
import pandas as pd

#import visualize

header = ['Gen', 'Pieces Placed', 'Computation Time [min]']
data = []
gen = 0
total_pp = 0
start_time = time.time()

def eval_genomes(genomes, config):
    """
    runs the simulation of the current population of
    agents and sets their fitness based on score.
    """
    global gen, total_pp, data
    gen += 1
    

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # environment that uses that network to play
    nets = []
    envs = []
    ge = []

    for _, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        envs.append(Tetris(True))
        ge.append(genome)

    nets = np.array(nets)
    envs = np.array(envs)
    ge = np.array(ge)

    # Run the simulation until all agents are dead
    rem = np.empty(len(envs))
    best_agent = None
    while len(envs) > 0:
        # Have each env take a step, by seeking all final states,
        # evaluating them all, inputting into NN and placing best state

        # Iterate through each agent
        for x, env in enumerate(envs):
            # Find all final_states and evaluate them
            states = env.get_final_states()
            evaluations = env.get_evaluations(states)
            
            # Pass the evaluation for each state into the NN
            outputs = [nets[x].activate(input) for input in evaluations]

            # Go to best scored state
            best_index = outputs.index(max(outputs))
            best_state = states[best_index]
            done = env.place_state(best_state)

            # Update fitness and remove if done
            ge[x].fitness = env.pieces_placed
            rem[x] = done
            total_pp += 1
        
        # Remove loser envs (rem == 0 - if done = False)
        best_agent = nets[0]
        nets = nets[rem == 0]
        envs = envs[rem == 0]
        ge = ge[rem == 0]

        rem = np.empty(len(envs))

    pickle.dump(best_agent, open("best.pickle", "wb"))
    
    if gen % 1 == 0: # Save milestones
        pickle.dump(best_agent, open("best.pickle_{}".format(gen), "wb"))    
        data.append([gen, total_pp, (time.time() - start_time) / 60])
        csv = pd.DataFrame(data, columns=header)
        csv.to_csv('Training_13.csv', index=False)
        
    

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

    # Run and indefinite amount of generations.
    winner = p.run(eval_genomes)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    #visualize.draw_net(config, winner, True)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)
    
    


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
