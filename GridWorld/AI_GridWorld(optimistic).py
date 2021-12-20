# Grid World: AI-controlled play

# Instructions:
#   Move up, down, left, or right to move the character. The 
#   objective is to find the key and get to the door
#
# Control:
#    arrows  : Merge up, down, left, or right
#    s       : Toggle slow play
#    a       : Toggle AI player
#    d       : Toggle rendering 
#    r       : Restart game
#    q / ESC : Quit



from GridWorld import GridWorld
import numpy as np
import pygame
from collections import defaultdict 
from matplotlib import pyplot as plt

# Testing parameters
steps = 0 # Number of steps during current iterations
iterations = 0 # Total number of iterations for current simulation

auto_testing = True
test_num = 1 # Upcoming test number
test_dens = 1000 # Test every # iterations
testing = False
xs, ys = 0,-1
T = np.array([[39,  0,  0,  0, 35, 34, 33, 32, 31,  0],
              [38, 37, 36, 37, 36,  0,  0,  0,  0,  0],
              [ 0,  0, 35,  0, 37, 38, 39,  0, 29, 28],
              [36, 35, 34,  0,  0,  0, 38,  0,  0, 27],
              [37,  0, 33, 34, 35, 36, 37,  0, 25, 26],
              [38,  0, 32,  0,  0,  0,  0,  0, 24,  0],
              [37,  0, 31,  0, 27, 26, 25, 24, 23, 22],
              [36,  0, 30, 29, 28,  0,  0,  0,  0, 21],
              [35,  0,  0,  0, 29,  0,  0, 16,  0, 20],
              [34, 33, 32, 31, 30,  0,  0, 17, 18, 19]])
steps_matrix = np.zeros(T.shape)
errors = np.array([])
data = np.array([])
N = 0
repeats = 0
total_iterations = np.array([])
failed = False
error = -1


# Initialize the environment
env = GridWorld()
env.reset()
x, y, has_key = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
slow = True
runai = True
render = True
done = False

# Game clock
clock = pygame.time.Clock()

# INSERT YOUR CODE HERE (1/2)
# Define data structure for q-table
# Here, we will use optimistic initialization and assume all state-actions 
#   have quality 0. This is optimistic, because each step yields reward -1 and 
#   only the key and door give positive rewards (50 and 100)
Q = defaultdict(lambda: [0., 0., 0., 0.])
# END OF YOUR CODE (1/2)

def plot(Y):
    Y = Y / N # Mean errors
    X = range(len(Y))
    # mean +- 1.96 * std / sqrt(n)
    ci = 1.96 * np.std(Y) / (N ** 0.5)
    plt.plot(X,Y)
    plt.xlabel("Iterations in thousands")
    plt.ylabel("Error")
    plt.fill_between(X, (Y-ci), (Y+ci), color='blue', alpha=0.1)
    plt.show()
    print("Plot for N={}".format(N))
    
    mean_ct = np.mean(total_iterations)
    ci_ct = 1.96 * np.std(total_iterations) / (len(total_iterations) ** 0.5)
    print("Mean convergence time: {}+-{}".format(mean_ct, ci_ct))

def get_new_start(xs, ys, testing):
    ys += 1 
    if ys == 10:
        ys = 0
        xs += 1
        if xs == 10: # Out of of bounds so done
            return 0, -1, False
    return xs, ys, testing

while not exit_program:
    if render:
        env.render()
    
    # Slow down rendering to 5 fps
    if slow and runai:
        clock.tick(5)
        
    # Automatic reset environment in AI mode
    if done and runai:
        if testing:
            if ys != -1:
                steps_matrix[xs,ys] = steps
                #print("Completed in {} steps".format(steps))
            
            xs, ys, testing = get_new_start(xs, ys, testing)
            
            while T[xs][ys] == 0: # Check that we aren't spawning in wall
                xs, ys, testing = get_new_start(xs, ys, testing)
            
            env.reset(ys, xs) # Set spawn point for next testiration
            
            
            if not testing or failed: # Done testing
                env.reset()
                e = 0.1
                ys = -1 # Some reason I must do this here again
                
                #print(steps_matrix)
                #print(steps_matrix-T)
                errors = np.append(errors, error)
                if error == sum(sum(steps_matrix-T)):
                    repeats += 1      
                    if repeats == 330:
                        print("330 repeats. Hard Resetting")
                error = sum(sum(steps_matrix-T))        
                
                
                if failed:
                    error = 1000000               
                
                print("Error: {}".format(error))
                #print(iterations, error)
                
                if error == 0 or repeats == 330 or failed: # Hard reset
                    Q = defaultdict(lambda: [0., 0., 0., 0.]) 
                    steps_matrix = np.zeros(T.shape)
                    print("Simulation done after {} iterations (in thousands)".format(iterations / test_dens))
                    iterations /= 1000.0
                    total_iterations = np.append(total_iterations, iterations)
                    iterations = 0
                    test_num = 1
                    repeats = 0
                    error = -1
                    failed = False
                    testing = False
                    steps = 0
                    
                    if len(data) < len(errors):
                        C = errors.copy()
                        C[:len(data)] += data
                    else:
                        C = data.copy()
                        C[:len(errors)] += errors
                        
                    data = C
                    #print("Data: {}".format(data))
                        
                    errors = np.array([])
                    N += 1
                    
                    plot(data)
                        
        else: env.reset()
        
        x, y, has_key = env.get_state()
        steps = 0
        
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_UP:
                action, action_taken = 'up', True
            if event.key == pygame.K_DOWN:
                action, action_taken  = 'down', True
            if event.key == pygame.K_RIGHT:
                action, action_taken  = 'right', True
            if event.key == pygame.K_LEFT:
                action, action_taken  = 'left', True
            if event.key == pygame.K_r:
                env.reset()   
            if event.key == pygame.K_d:
                render = not render
            if event.key == pygame.K_s:
                slow = not slow
            if event.key == pygame.K_a:
                runai = not runai
                clock.tick(5)
            if event.key == pygame.K_i:
                print("Iterations: {}".format(iterations))
            if event.key == pygame.K_k:
                print("Steps: {}".format(steps))
    
    # AI controller (enable/disable by pressing 'a')
    if runai:
        # INSERT YOUR CODE HERE (2/2)
        #
        # Implement a Grid World AI (q-learning): Control the person by 
        # learning the optimal actions through trial and error
        #
        # The state of the environment is available in the variables
        #    x, y     : Coordinates of the person (integers 0-9)
        #    has_key  : Has key or not (boolean)
        #
        # To take an action in the environment, use the call
        #    (x, y, has_key), reward, done = env.step(action)
        #
        #    This gives you an updated state and reward as well as a Boolean 
        #    done indicating if the game is finished. When the AI is running, 
        #    the game restarts if done=True

        # 1. choose an action
        x = env.x
        y = env.y
        q_current = Q[(x,y,has_key)]    
        top_actions = np.argwhere(q_current == np.amax(q_current)).flatten()
        action_num = np.random.choice(top_actions)
        #action_num =  np.argmax(q_current)
        action = actions[action_num]  
        
        while True: # Break out of loop if valid action
            if action =='left':
                if x != 0:
                    break
            elif action =='right':
                if x != 9:
                    break
            elif action =='up':
                if y != 0:
                    break
            elif action =='down':
                if y != 9:
                    break
            
            # Not valid action, so punished and new action found
            q_current[action_num] = -100
            top_actions = np.argwhere(q_current == np.amax(q_current)).flatten()
            action_num = np.random.choice(top_actions)
            action = actions[action_num]
            
        
        # 2. step the environment
        (x, y, has_key), reward, done = env.step(action)
        steps += 1
        
        if not testing:
            # 2.1 Check if time to test (but not while testing)
            iterations += 1
            if auto_testing and test_dens * test_num == iterations:
                # Test and force reset
                test_num += 1
                testing = True
                e = 0
                done = True
        
        if testing:
            # 2.2 Check if test failed
            if steps > 1000:
                    testing = False
                    print("FAIL")
                    done = True
        
        
        # 3. update q table
        if not testing:
            q_next = Q[(x, y, has_key)]
            q_current[action_num] = reward + 0.9*np.max(q_next)

        
        

        # END OF YOUR CODE (2/2)
    
    # Human controller        
    else:
        if action_taken:
            (x, y, has_key), reward, done = env.step(action)
            action_taken = False


env.close()
