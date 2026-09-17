"""无窗口验证奖励和训练数据流，使用真实 NetworkX 搜索。"""
import importlib.util
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np  # 在临时替换 sys.modules 前加载，避免测试间重复导入。


def load_module(name, filename, mocks):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, mocks):
        spec.loader.exec_module(module)
    return module


pygame = ModuleType('pygame')
pygame.event = SimpleNamespace(get=lambda: [])
pygame_locals = ModuleType('pygame.locals')
pygame_locals.QUIT = 256
env = load_module('reward_test_env', 'snake_env.py',
                  {'pygame': pygame, 'pygame.locals': pygame_locals})
P = env.Position


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.game = env.Game.__new__(env.Game)
        self.game.space_w = self.game.space_h = 5
        rows = ['1111111'] + ['1000001'] * 5 + ['1111111']
        self.game.wall = SimpleNamespace(map=[list(row) for row in rows])
        self.game.path_graph = self.game.build_path_graph()
        self.game.berry = SimpleNamespace(position=P(4, 2))
        self.game.snake = env.Snake.__new__(env.Snake)
        self.game.snake.blocks = [P(2, 2), P(1, 2)]
        self.game.snake.current_direction = env.Direction.right
        self.game.prev_blocks = tuple(self.game.snake.blocks)
        self.game.score = self.game.total_step = self.game.frame = 0
        self.game.draw = lambda: None
        self.game.Clock = SimpleNamespace(tick=lambda _: None)
        self.game.position_berry = lambda: setattr(self.game.berry, 'position', P(5, 5))

    def test_body_detour(self):
        self.assertEqual(self.game.food_path_distance([P(2, 2), P(3, 2)]), 4)

    def test_wall_detour(self):
        self.game.wall.map[2][3] = '1'
        self.game.path_graph = self.game.build_path_graph()
        self.assertEqual(self.game.food_path_distance([P(2, 2)]), 4)

    def test_unreachable_stays_finite_and_shaping_is_bounded(self):
        self.game.snake.blocks = [P(2, 2)] + [P(3, y) for y in range(1, 6)]
        self.assertEqual(self.game.food_path_distance(self.game.snake.blocks), 25)
        self.assertAlmostEqual(self.game.compute_reward(False), -0.2)
        self.game.prev_blocks = tuple(self.game.snake.blocks)
        self.assertAlmostEqual(self.game.compute_reward(False), -0.1)

    def test_search_matches_bfs_on_random_obstacles(self):
        rng = random.Random(42)
        for _ in range(100):
            body = rng.sample([p for p in self.game.path_graph
                               if p not in (P(2, 2), self.game.berry.position)],
                              rng.randrange(15))
            graph = self.game.path_graph.copy()
            graph.remove_nodes_from(body)
            try:
                expected = nx.shortest_path_length(graph, P(2, 2), self.game.berry.position)
            except nx.NetworkXNoPath:
                expected = len(self.game.path_graph)
            self.assertEqual(self.game.food_path_distance([P(2, 2)] + body), expected)

    def test_step_returns_progress_reward_and_keeps_previous_body(self):
        previous = tuple(self.game.snake.blocks)
        reward, done, score = self.game.play_step(0)
        self.assertAlmostEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(score, 0)
        self.assertEqual(self.game.prev_blocks, previous)

    def test_step_returns_food_reward_even_after_respawn(self):
        self.game.berry.position = P(3, 2)
        self.assertEqual(self.game.play_step(0), (10.0, False, 1))
        self.assertEqual(self.game.berry.position, P(5, 5))

    def test_step_returns_death_penalty(self):
        self.game.snake.blocks = [P(1, 2), P(2, 2)]
        self.game.snake.current_direction = env.Direction.left
        self.assertEqual(self.game.play_step(0), (-10, True, 0))

    def test_training_keeps_rewards_and_uses_reset_state(self):
        transitions = []
        actions = []

        class FakeGame:
            nS, nA = 34, 3

            def __init__(self):
                self.state = 'initial'
                self.steps = iter([(10, False, 1), (-10, True, 1), (-10, True, 3)])

            def get_state(self):
                return self.state

            def play_step(self, action):
                self.state = 'terminal-or-next'
                return next(self.steps)

            def reset(self):
                self.state = 'reset'

        class FakeAgent:
            def __init__(self, *args):
                self.n_game = 0
                self.best_reward = 99
                self.trainer = SimpleNamespace(save=lambda **kwargs: None)

            def get_action(self, state, n_game):
                actions.append(state)
                return 0

            def remember(self, *args):
                transitions.append(args)

            def train_long_memory(self, **kwargs):
                pass

        training = load_module('reward_test_training', 'train.py', {
            'snake_env': SimpleNamespace(Game=FakeGame),
            'Agent': SimpleNamespace(Agent=FakeAgent),
            'helper': SimpleNamespace(plot=lambda *args: None),
        })
        training.train()
        self.assertEqual([t[2] for t in transitions], [10, -10, -10])
        self.assertEqual(actions, ['initial', 'terminal-or-next', 'reset'])


if __name__ == '__main__':
    unittest.main()
