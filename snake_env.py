import pygame
import random
from collections import namedtuple
from pygame.locals import QUIT
import numpy as np
import networkx as nx
import sys

Position = namedtuple('Point', 'x, y')

class Direction:
    right = 0
    left = 1
    up = 2
    down = 3

class Snake:

    def __init__(self, block_size):
        self.blocks = []
        # 定义蛇头蛇尾生成的初始位置
        self.blocks.append(Position(7, 7))
        self.blocks.append(Position(7, 6))

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

    def handle_action(self, action):
        clock_wise = [Direction.right, Direction.down, Direction.left, Direction.up]
        idx = clock_wise.index(self.current_direction)
        try:
            if action == 0:
                new_dir = clock_wise[idx]
            elif action == 1:
                next_idx = (idx + 1) % 4
                new_dir = clock_wise[next_idx]
            elif action == 2:
                next_idx = (idx - 1) % 4
                new_dir = clock_wise[next_idx]
            self.current_direction = new_dir
        except:
            print(action,type(action))
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
        self.map = self.load_map('./assets/map15x15.txt')
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

    def __init__(self, Width=240, Height=240):
        pygame.init()
        self.block_size = 16
        # 定义窗口大小
        self.w, self.h = (Width, Height)
        # 定义活动空间
        self.space_w = self.w // self.block_size - 2
        self.space_h = self.h // self.block_size - 2

        self.surface = pygame.display.set_mode((self.w, self.h))
        self.font = pygame.font.SysFont(None, 16)
        self.Clock = pygame.time.Clock()
        self.snake = Snake(self.block_size)
        self.berry = Berry(self.block_size)
        self.wall = Wall(self.block_size)
        self.path_graph = self.build_path_graph()

        # 输入的特征值
        self.nS = 34
        # 输出值,对应直走、向左拐、向右拐
        self.nA = 3
        # 重置
        self.reset()

    def reset(self):

        '''
        局内属性
        与之前人操作的一些属性相同，每次重置游戏这些属性都会重置
        该函数以外的为全局属性
        '''

        self.score = 0
        self.frame = 0
        self.snake = Snake(self.block_size)
        self.position_berry()
        self.total_step = 0
        self.reward = 0.0
        self.prev_head = self.snake.blocks[0]
        self.prev_blocks = tuple(self.snake.blocks)

    def build_path_graph(self):
        """缓存地图的四邻接图，蛇身在每次搜索时动态过滤。"""
        graph = nx.Graph()
        for y in range(1, self.space_h + 1):
            for x in range(1, self.space_w + 1):
                point = Position(x, y)
                if not self.head_hit_wall(point):
                    graph.add_node(point)
        for point in list(graph):
            for neighbor in (Position(point.x + 1, point.y),
                             Position(point.x, point.y + 1)):
                if neighbor in graph:
                    graph.add_edge(point, neighbor)
        return graph

    def food_path_distance(self, blocks):
        """当前蛇身视为静态障碍；不可达时返回有限距离，避免 NaN。"""
        head = blocks[0]
        body = set(blocks[1:])
        graph = nx.subgraph_view(self.path_graph,
                                 filter_node=lambda point: point not in body)
        unreachable = len(self.path_graph)
        if head not in graph or self.berry.position not in graph:
            return unreachable
        try:
            return nx.astar_path_length(
                graph, head, self.berry.position,
                heuristic=lambda a, b: abs(a.x - b.x) + abs(a.y - b.y))
        except nx.NetworkXNoPath:
            return unreachable

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
            return True
        else:
            self.snake.blocks.pop()
            return False

    def head_hit_body(self, position=None):
        if position is None:
            position = self.snake.blocks[0]
        if position in self.snake.blocks[1:]:
            return True
        return False

    def head_hit_wall(self, position=None):
        if position is None:
            position = self.snake.blocks[0]
        if position.y >= 14 or position.x >= 14 or position.x <= 0 or position.y <= 0:
            return True
        if self.wall.map[position.y][position.x] == '1':
            return True
        return False

    def compute_reward(self, ate_berry):
        reward = -0.1

        if ate_berry:
            reward = 10.0
            return reward

        prev_dist = self.food_path_distance(self.prev_blocks)
        curr_dist = self.food_path_distance(self.snake.blocks)
        # 蛇身移动可能让路径长度突变，限制塑形奖励以保留终局奖励的主导作用。
        progress = max(-1, min(1, prev_dist - curr_dist))
        reward += progress * 0.1

        return reward
    def draw_data(self):
        text = f"score = {self.score}"
        text_img = self.font.render(text, 1, Game.WE)
        text_rect = text_img.get_rect(centerx=self.surface.get_width() / 2, top=32)
        self.surface.blit(text_img, text_rect)

    def draw(self):
        self.surface.fill(Game.BK)
        self.wall.draw(self.surface)
        self.berry.draw(self.surface)
        self.snake.draw(self.surface, self.frame)
        self.draw_data()
        pygame.display.update()

    # main loop
    def play_step(self, action):
        game_over = False
        self.reward = 0.0

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()  # 整个程序运行结束

        self.total_step += 1
        self.frame = (self.frame + 1) % 2

        # 用于保存前一次蛇的位置
        self.prev_head = self.snake.blocks[0]
        self.prev_blocks = tuple(self.snake.blocks)

        self.snake.handle_action(action)
        ate_berry = self.berry_collision()

        if (self.head_hit_body() or
            self.head_hit_wall() or
            self.total_step > 100*len(self.snake.blocks)):   # 检测一直在绕圈不吃berry
            game_over = True
            # 惩罚机制
            self.reward = -10
            return self.reward, game_over, self.score

        self.reward = self.compute_reward(ate_berry)

        self.draw()
        self.Clock.tick(60)
        return self.reward, game_over, self.score


    def get_state(self):

        '''
        用于获取输入网络的状态信息矩阵
        '''

        # 获取头及其上下左右的状态
        head = self.snake.blocks[0]
        point_l = Position(head.x - 1, head.y)
        point_r = Position(head.x + 1, head.y)
        point_u = Position(head.x, head.y - 1)
        point_d = Position(head.x, head.y + 1)

        # 危险信号(周围一格的范围内的所有存在的危险，用于紧急避障)
        danger_1b_r = self.head_hit_body(point_r)
        danger_1b_l = self.head_hit_body(point_l)
        danger_1b_u = self.head_hit_body(point_u)
        danger_1b_d = self.head_hit_body(point_d)
        danger_1w_r = self.head_hit_wall(point_r)
        danger_1w_l = self.head_hit_wall(point_l)
        danger_1w_u = self.head_hit_wall(point_u)
        danger_1w_d = self.head_hit_wall(point_d)

        # 获取从头一直延伸到地图边缘的所有状态  在大地图中这些状态不够精细
        points_l = [Position(i, head.y) for i in range(1, head.x)]
        points_r = [Position(i, head.y) for i in range(head.x + 1, self.space_w)]
        points_u = [Position(head.x, i) for i in range(1, head.y)]
        points_d = [Position(head.x, i) for i in range(head.y + 1, self.space_h)]

        # 危险信号(以头为中心的上左下右延伸到边界检测危险，用于走位规划)
        danger_b_l = np.any(np.array([self.head_hit_body(point) for point in points_l]))
        danger_b_r = np.any(np.array([self.head_hit_body(point) for point in points_r]))
        danger_b_u = np.any(np.array([self.head_hit_body(point) for point in points_u]))
        danger_b_d = np.any(np.array([self.head_hit_body(point) for point in points_d]))

        dir_l = self.snake.current_direction == Direction.left
        dir_r = self.snake.current_direction == Direction.right
        dir_u = self.snake.current_direction == Direction.up
        dir_d = self.snake.current_direction == Direction.down

        berry_l = self.berry.position.x < head.x
        berry_r = self.berry.position.x > head.x
        berry_u = self.berry.position.y < head.y
        berry_d = self.berry.position.y > head.y

        # 距离特征
        dist_berry_x = self.berry.position.x - head.x
        dist_berry_y = self.berry.position.y - head.y
        dist_wall_left = head.x
        dist_wall_right = self.space_w - head.x - 1
        dist_wall_up = head.y
        dist_wall_down = self.space_h - head.y - 1

        # 离最近身体块距离（上下左右）
        def distance_to_body(dx, dy):
            step = 0
            x, y = head.x + dx, head.y + dy
            max_dist = max(self.space_w, self.space_h)

            while 0 <= x < self.space_w and 0 <= y < self.space_h:
                step += 1
                if Position(x, y) in self.snake.blocks[1:]:
                    return step / max_dist
                x += dx
                y += dy
            return 1.0  # 如果没有碰到身体，就返回到墙的距离

        dist_body_l = distance_to_body(-1, 0)
        dist_body_r = distance_to_body(1, 0)
        dist_body_u = distance_to_body(0, -1)
        dist_body_d = distance_to_body(0, 1)

        def distance_to_berry(dx, dy):
            """
            从蛇头沿着方向(dx, dy)计算到食物的距离（格子数）
            """
            step = 0
            x, y = head.x + dx, head.y + dy
            max_dist = max(self.space_w, self.space_h)

            while 0 <= x < self.space_w and 0 <= y < self.space_h:
                step += 1
                if Position(x, y) == self.berry.position:
                    return step / max_dist
                x += dx
                y += dy
            return 1.0  # 如果沿着方向没有遇到食物，返回 None 或者可以返回一个大数

        dist_berry_l = distance_to_berry(-1, 0)
        dist_berry_r = distance_to_berry(1, 0)
        dist_berry_u = distance_to_berry(0, -1)
        dist_berry_d = distance_to_berry(0, 1)

        # 输入网络的状态信息矩阵(共20项)
        state = [
            danger_1b_r, danger_1b_l, danger_1b_u, danger_1b_d,
            danger_1w_r, danger_1w_l, danger_1w_u, danger_1w_d,
            danger_b_r, danger_b_l, danger_b_u, danger_b_d,
            dir_l, dir_r, dir_u, dir_d,
            berry_l, berry_r, berry_u, berry_d,
            dist_berry_x, dist_berry_y, dist_wall_left, dist_wall_right,
            dist_wall_up, dist_wall_down, dist_body_l, dist_body_r,
            dist_body_u, dist_body_d, dist_berry_l, dist_berry_r,
            dist_berry_u, dist_berry_d
        ]

        return np.array(state, dtype=float)
