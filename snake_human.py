import pygame
import random
from collections import namedtuple
from pygame.locals import K_RIGHT, K_LEFT, K_UP, K_DOWN, QUIT

Position = namedtuple('Point', 'x, y')

class Direction:
    right = 0
    left = 1
    up = 2
    down = 3

class Snake:

    def __init__(self,block_size):
        self.blocks = []
        # 定义蛇头蛇尾生成的初始位置
        self.blocks.append(Position(20, 15))
        self.blocks.append(Position(19, 15))

        self.block_size = block_size
        # 定义蛇头的初始方向
        self.current_direction = Direction.right
        # 提取材质
        self.image = pygame.image.load('./assets/textures/snake.png')

    def move(self):
        if (self.current_direction == Direction.right):
            movesize = (1, 0)
        elif (self.current_direction == Direction.left):
            movesize = (-1, 0)
        elif (self.current_direction == Direction.up):
            movesize = (0, -1)
        else:
            movesize = (0, 1)
        head = self.blocks[0]
        new_head = Position(head.x + movesize[0], head.y + movesize[1])
        self.blocks.insert(0, new_head)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        if keys[K_RIGHT] and self.current_direction != Direction.left:
            self.current_direction = Direction.right
        elif keys[K_LEFT] and self.current_direction != Direction.right:
            self.current_direction = Direction.left
        elif keys[K_UP] and self.current_direction != Direction.down:
            self.current_direction = Direction.up
        elif keys[K_DOWN] and self.current_direction != Direction.up:
            self.current_direction = Direction.down
        self.move()

    def draw(self, surface, frame):
        for idx, block in enumerate(self.blocks):
            position = (block.x * self.block_size, block.y * self.block_size)
            if idx == 0:
                src = ((self.current_direction * 2 + frame) * self.block_size,0,
                       self.block_size,self.block_size)
            else:
                src = (8 * self.block_size, 0, self.block_size, self.block_size)
            surface.blit(self.image, position, src)

class Berry:

    def __init__(self, block_size):
        self.block_size = block_size
        self.image = pygame.image.load('./assets/textures/berry.png')
        self.position = Position(1, 1)

    def draw(self, surface):
        rect = self.image.get_rect()
        rect.left = self.position.x * self.block_size
        rect.top = self.position.y * self.block_size
        surface.blit(self.image, rect)

class Wall:

    def __init__(self, block_size):
        self.block_size = block_size
        self.map = self.load_map('./assets/map.txt')
        self.image = pygame.image.load('./assets/textures/wall.png')


    def load_map(self, fileName):
        with open(fileName, 'r') as mf:
            content = mf.readlines()
            content = [list(line.strip()) for line in content]
        return content

    def draw(self, surface):
        for row, line in enumerate(self.map):
            for col, value in enumerate(line):
                if value == '1':
                    position = (col * self.block_size, row * self.block_size)
                    surface.blit(self.image, position)

class Game:
    WE = (255, 255, 255)
    BK = (0, 0, 0)

    def __init__(self, Width=640, Height=480):
        pygame.init()
        self.block_size = 16
        # 定义窗口大小
        self.w, self.h = (Width, Height)
        # 定义活动空间
        self.space_w = self.w // self.block_size - 2
        self.space_h = self.h // self.block_size - 2

        self.surface = pygame.display.set_mode((self.w, self.h))
        self.score, self.frame = 0, 0
        self.running = True
        self.Clock = pygame.time.Clock()
        self.fps = 5
        self.font = pygame.font.SysFont(None, 32)
        self.snake = Snake(self.block_size)
        self.berry = Berry(self.block_size)
        self.wall = Wall(self.block_size)
        self.position_berry()

    def position_berry(self):
        bx = random.randint(1, self.space_w)
        by = random.randint(1, self.space_h)
        self.berry.position = Position(bx, by)
        if self.berry.position in self.snake.blocks:
            self.position_berry()

    def berry_collision(self):
        head = self.snake.blocks[0]
        if (head.x == self.berry.position.x and
                head.y == self.berry.position.y):
            self.position_berry()
            self.score += 1
        else:
            self.snake.blocks.pop()

    def head_hit_body(self):
        head = self.snake.blocks[0]
        if head in self.snake.blocks[1:]:
            return True
        return False

    def head_hit_wall(self):
        head = self.snake.blocks[0]
        if self.wall.map[head.y][head.x] == '1':
            return True
        return False

    def draw_data(self):
        text = f"score = {self.score}"
        text_img = self.font.render(text, 1, Game.WE)
        text_rect = text_img.get_rect(centerx=self.surface.get_width() / 2, top=32)
        self.surface.blit(text_img, text_rect)

    def draw(self):
        self.surface.fill(Game.BK)
        self.wall.draw(self.surface)
        self.berry.draw(self.surface)
        self.snake.draw(self.surface,self.frame)
        self.draw_data()
        pygame.display.update()

    # main loop
    def play(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

            self.frame = (self.frame + 1) % 2
            self.snake.handle_input()
            self.berry_collision()
            if self.head_hit_body() or self.head_hit_wall():
                print('Final Score', self.score)
                self.running = False

            self.draw()
            self.Clock.tick(self.fps)

        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.play()