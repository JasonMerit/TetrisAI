# Grid World game

# Import libraries used for this program
 
import pygame
import numpy as np



class Tetris():    
    # Rendering?
    rendering = False
    
    # Rendering Dimensions
    screenSize = 600
    cellSize = 20
    offset = 100

    # Colors
    yellow = (236,226,157)
    red = (180,82,80)
    cyan = (105,194,212)
    blue = (75,129,203,)
    pink = (205,138,206)
    orange = (211,160,103)
    green = (75,129,203)
    badColor = (192, 30, 30)
    black = (34,34,34)
    grey = (184,184,184)
    
    def __init__(self, state=None):
        pygame.init()
        self.reward = 0
        self.x, self.y, self.board, self.score = self.new_game()
        
    def get_state(self):
        return (self.x, self.y)
            
    def step(self, action):
        # Move piece
        if not self.game_over(self.y, self.board):
            self.x, self.y, self.board, self.score = self.move(self.x, self.y, self.board, self.score, action)
        
               
        return (self.x, self.y, done)
    
    def tick(self):        
        # Let piece fall 
        if not self.game_over(self.y, self.board):
            if y == 23 or self.board[y+1,x] != 0:
                # Hit floor or other piece
                # Set PLACED to true
                pass
            else:
                ny = y + 1 #Drop down
        
        self.y = ny
        
        # return observation, reward, done
        done = self.game_over(self.y, self.board) 
        return (self.x, self.y, done)
        
    def render(self):
        if not self.rendering:
            self.init_render()
                 
        # Clear the screen
        self.screen.fill(self.black)
        
        # Draw board
        border = 0.5
        pygame.draw.rect(self.screen, self.grey, pygame.Rect(150-2,100-2,200+3,480+3))
        for i in range(len(self.board[0])):
            for j in range(len(self.board)):
                val = self.board[j,i]
                col = self.red if val != 0 else self.black
                pygame.draw.rect(self.screen, col, pygame.Rect(150+self.cellSize*i+border,100+self.cellSize*j+border,self.cellSize-2*border,self.cellSize-2*border))
               


        text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        self.screen.blit(text, (790-text.get_width(), 10))

        # Draw game over or you won       
        if self.game_over(self.y, self.board):
            msg = 'Game over!'
            col = self.badColor
            text = self.bigfont.render(msg, True, col)
            textpos = text.get_rect(centerx=self.background.get_width()/2)
            textpos.top = 300
            self.screen.blit(text, textpos)

        # Display
        pygame.display.flip()

    def reset(self):
        self.x, self.y, self.board, self.score = self.new_game()

    def close(self):
        pygame.quit()
                 
    def init_render(self):
        
        self.screen = pygame.display.set_mode([self.screenSize, self.screenSize])
        pygame.display.set_caption('Tetris')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)
           
    def game_over(self, y, board):
        #if np.any(board[0]):
            #return True
        
        return False
        
    def move(self, x, y, board, score, action):
        nx, ny = x, y
        
        if action=='left':
            if nx>0:
                nx = x - 1
        elif action=='right':
            if nx<9:
                nx = x + 1
        elif action=='down':
            if y != 23:
                if board[y+1,x] == 0:
                    ny = y + 1 #Drop down
        elif action=="A":
            nx, ny = self.rotate(True)
        elif action=="B":
            nx, ny = self.rotate(False)
        else:
            pass

        # Update position
        board[y, x] = 0
        board[ny, nx] = 1
        self.x, self.y = nx, ny
                     
        return (nx, ny, board, score)
       
    def new_game(self):
        board = np.zeros([24,10])
        score = 0
        self.x, self.y = 4, 0
        board[self.y, self.x] = 1
        return (self.x, self.y, board, score)


# Initialize the environment
env = Tetris()
env.reset()
x, y = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
slow = True
runai = False
render = True
done = False

clock = pygame.time.Clock()

while not exit_program:
    clock.tick(40)
        
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_RIGHT:
                action, action_taken = "right", True
            if event.key == pygame.K_LEFT:
                action, action_taken = "left", True
            if event.key == pygame.K_UP:
                action, action_taken = "up", True
            if event.key == pygame.K_DOWN or event.key == pygame.K_SPACE:
                action, action_taken = "down", True
    
    # AI controller
    if runai:
        pass
    
    # Human controller
    else:
        if action_taken:
            x, y, done = env.step(action)
            action_taken = False
            
    # Process game tick
    env.tick()
            
    if render:
        env.render()
    
    

env.close()
