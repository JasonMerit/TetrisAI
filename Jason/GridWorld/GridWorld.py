# Grid World game

# Import libraries used for this program
 
import pygame
import numpy as np

#%%

class GridWorld():    
    # Rendering?
    rendering = False
    
    # Images
    filenames = ['person.png', 'key.png', 'door.png', 'death.png']
    images = [pygame.image.load(file) for file in filenames]

    # Colors
    goodColor = (30, 192, 30)
    badColor = (192, 30, 30)
    pathColor = (225, 220, 225)
    wallColor = (157, 143, 130)
    
    def __init__(self, state=None):
        pygame.init()
        self.reward = 0
        if state is None:
            self.x, self.y, self.has_key, self.board, self.score = self.new_game(-1,-1)            
        else:
            x, y, has_key, board, score = state
            self.x, self.y, self.has_key, self.board, self.score = x, y, has_key, board.copy(), score
    
    def get_state(self):
        return (self.x, self.y, self.has_key)
            
    def step(self, action):
        # Move character
        if not self.game_over(self.x, self.y, self.has_key, self.board):
            self.x, self.y, self.has_key, self.board, self.score, self.reward = self.move(self.x, self.y, self.has_key, self.board, self.score, action)
        
        # return observation, reward, done
        done = self.game_over(self.x, self.y, self.has_key, self.board)        
        return ((self.x, self.y, self.has_key), self.reward, done)
        
    def render(self):
        if not self.rendering:
            self.init_render()
                 
        # Clear the screen
        self.screen.fill((187,173,160))
        
        border = 3
        pygame.draw.rect(self.screen, (187,173,160), pygame.Rect(100,0,600,600))
        for i in range(10):
            for j in range(10):
                val = self.board[i,j]
                col = self.wallColor if val & 8 else self.pathColor
                pygame.draw.rect(self.screen, col, pygame.Rect(100+60*i+border,60*j+border,60-2*border,60-2*border))
                if val>0:
                    x = 105 + 60*i
                    y = 5 + 60*j
                    if val & 4:
                        self.screen.blit(self.images[2], (x, y))
                    if val & 2:
                        self.screen.blit(self.images[1], (x, y))
                    if val & 1:
                        if self.game_over(self.x, self.y, self.has_key, self.board) and not self.won(self.x, self.y, self.has_key, self.board):
                            self.screen.blit(self.images[3], (x, y))
                        else:
                            self.screen.blit(self.images[0], (x, y))
        text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        self.screen.blit(text, (790-text.get_width(), 10))
        #text = self.scorefont.render("Jason is hot", True, (0,0,0))
        #self.screen.blit(text, (790-text.get_width(), 20))
        
        # Draw game over or you won       
        if self.game_over(self.x, self.y, self.has_key, self.board):
            if self.won(self.x, self.y, self.has_key, self.board):
                msg = 'Congratulations!'
                col = self.goodColor
            else:
                msg = 'Game over!'
                col = self.badColor
            text = self.bigfont.render(msg, True, col)
            textpos = text.get_rect(centerx=self.background.get_width()/2)
            textpos.top = 300
            self.screen.blit(text, textpos)

        # Display
        pygame.display.flip()

    def reset(self, sx = -1, sy = -1):
        self.x, self.y, self.has_key, self.board, self.score = self.new_game(sx,sy)

    def close(self):
        pygame.quit()
                 
    def init_render(self):
        self.screen = pygame.display.set_mode([800, 600])
        pygame.display.set_caption('Grid World')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)
           
    def game_over(self, x, y, has_key, board):
        # Are we on a death square?
        if board[x,y] & 8:
            return True
        
        # Are we on the door with the key?
        if board[x,y] & 4 and not np.any(board & 2):            
            return True
        
        return False
    
    def won(self, x, y, has_key, board):
        # Are we on the door with the key?
        if board[x,y] & 4 and not np.any(board & 2):            
            return True
        
        return False
        
        
    def move(self, x, y, has_key, board, score, direction='left'):
        newx, newy = x, y
        if direction=='left':
            if x>0:
                newx = x-1
        elif direction=='right':
            if x<9:
                newx = x+1
        elif direction=='up':
            if y>0:
                newy = y-1                
        elif direction=='down':
            if y<9:
                newy = y+1
        
        # Jason was here
        reward = 0
        
        # Update position
        board[x,y] -= 1
        board[newx, newy] += 1
        self.x, self.y = newx, newy
        
        # Take key
        if board[newx, newy] & 2:
            board[newx, newy] -= 2
            reward = 50
            has_key = True
        
        # On door with key?
        if board[newx, newy] & 4 and not np.any(board & 2):
            reward = 100
        
        # On death?
        if board[newx, newy] & 8:
            reward = -100

        score += reward                        
        return (newx, newy, has_key, board, score, reward)
       
    def new_game(self, sx, sy):
        board = np.loadtxt('board.txt', dtype=int).T
        if board.shape != (10,10) or np.sum(board==2) != 1 or np.sum(board==4) != 1:
            raise Exception('board.txt corrupt')
            
        if sx == -1 and sy == -1:
            start_x, start_y = np.where(board == 0)
            i = np.random.randint(len(start_x))
            x, y = start_x[i], start_y[i]
            
        else:
            x, y = sx, sy
        
        board[x, y] = 1
        score = 0
        has_key = False
        return (x, y, has_key, board, score)




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
from collections import defaultdict 

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

while not exit_program:
    if render:
        env.render()
    
    # Slow down rendering to 5 fps
    if slow and runai:
        clock.tick(5)
        
    # Automatic reset environment in AI mode
    if done and runai:
        env.reset()
        x, y, has_key = env.get_state()
        
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
        q_current = Q[(x,y,has_key)]    
        action_num =  np.argmax(q_current)
        action = actions[action_num]  
        # 2. step the environment
        (x, y, has_key), reward, done = env.step(action)
        # 3. update q table
        q_next = Q[(x, y, has_key)]
        q_current[action_num] = reward + 0.9*np.max(q_next)

        # END OF YOUR CODE (2/2)
    
    # Human controller        
    else:
        if action_taken:
            (x, y, has_key), reward, done = env.step(action)
            action_taken = False


env.close()
