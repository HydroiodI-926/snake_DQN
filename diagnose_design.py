"""对贪吃蛇 DQN 的观测、奖励和回放策略做快速、无训练诊断。

本脚本不修改环境或模型，也不会执行训练。它验证三个问题：
1. 按 ``abs(reward)`` 采样时，各类经验实际得到的相对权重；
2. 当前 34 维观测是否会把需要不同决策的棋盘压成同一个状态；
3. 符合 DQN 折扣因子的 potential-based shaping 是否满足望远镜求和性质。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import networkx as nx
import numpy as np

try:
    from snake_env import Direction, Game, Position
except ModuleNotFoundError as error:
    if error.name != "pygame":
        raise
    # 诊断不创建 Game/Snake/Wall 实例，因此只需满足模块导入所需的名称。
    pygame_stub = ModuleType("pygame")
    pygame_stub.event = SimpleNamespace(get=lambda: [])
    pygame_locals_stub = ModuleType("pygame.locals")
    pygame_locals_stub.QUIT = 256
    sys.modules["pygame"] = pygame_stub
    sys.modules["pygame.locals"] = pygame_locals_stub
    from snake_env import Direction, Game, Position


ROOT = Path(__file__).resolve().parent
BOARD_SIZE = 13
ACTION_NAMES = {0: "直行", 1: "右转", 2: "左转"}


@dataclass(frozen=True)
class AliasingWitness:
    observation: tuple[float, ...]
    first_body: tuple[Position, ...]
    first_actions: frozenset[int]
    second_body: tuple[Position, ...]
    second_actions: frozenset[int]


def make_game(blocks: tuple[Position, ...], food: Position) -> Game:
    """绕过 Pygame 初始化，只构造 get_state 所需的最小环境。"""
    game = Game.__new__(Game)
    game.space_w = game.space_h = BOARD_SIZE
    rows = (ROOT / "assets" / "map15x15.txt").read_text(encoding="utf-8").splitlines()
    game.wall = SimpleNamespace(map=[list(row) for row in rows])
    game.snake = SimpleNamespace(
        blocks=list(blocks), current_direction=Direction.right
    )
    game.berry = SimpleNamespace(position=food)
    return game


def observation_of(blocks: tuple[Position, ...], food: Position) -> tuple[float, ...]:
    return tuple(make_game(blocks, food).get_state().tolist())


def action_destination(head: Position, direction: int, action: int) -> Position:
    clockwise = [Direction.right, Direction.down, Direction.left, Direction.up]
    index = clockwise.index(direction)
    if action == 1:
        index = (index + 1) % 4
    elif action == 2:
        index = (index - 1) % 4
    next_direction = clockwise[index]
    dx, dy = {
        Direction.right: (1, 0),
        Direction.left: (-1, 0),
        Direction.up: (0, -1),
        Direction.down: (0, 1),
    }[next_direction]
    return Position(head.x + dx, head.y + dy)


def optimal_first_actions(
    blocks: tuple[Position, ...], food: Position
) -> frozenset[int]:
    """按当前项目的“整条蛇身是静态障碍”假设求 A* 最优首动作。"""
    game = make_game(blocks, food)
    game.path_graph = game.build_path_graph()
    body = set(blocks[1:])
    best_distance = float("inf")
    best_actions: set[int] = set()

    for action in range(3):
        destination = action_destination(blocks[0], Direction.right, action)
        if destination in body or game.head_hit_wall(destination):
            continue
        graph = game.path_graph.copy()
        graph.remove_nodes_from(body)
        try:
            distance = nx.astar_path_length(
                graph,
                destination,
                food,
                heuristic=lambda a, b: abs(a.x - b.x) + abs(a.y - b.y),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if distance < best_distance:
            best_distance = distance
            best_actions = {action}
        elif distance == best_distance:
            best_actions.add(action)
    return frozenset(best_actions)


def random_connected_body(
    rng: random.Random, length: int, food: Position
) -> tuple[Position, ...] | None:
    """从固定蛇头和脖子反向生长一条合法、自不相交的蛇身。"""
    blocks = [Position(7, 7), Position(6, 7)]
    occupied = set(blocks)
    while len(blocks) < length:
        tail = blocks[-1]
        candidates = [
            Position(tail.x + 1, tail.y),
            Position(tail.x - 1, tail.y),
            Position(tail.x, tail.y + 1),
            Position(tail.x, tail.y - 1),
        ]
        candidates = [
            point
            for point in candidates
            if 1 <= point.x <= BOARD_SIZE
            and 1 <= point.y <= BOARD_SIZE
            and point not in occupied
            and point != food
        ]
        if not candidates:
            return None
        point = rng.choice(candidates)
        blocks.append(point)
        occupied.add(point)
    return tuple(blocks)


def find_aliasing_witness(seed: int = 20260920, attempts: int = 30_000) -> AliasingWitness:
    """寻找观测完全相同、A* 最优首动作却互斥的两个合法蛇身。"""
    rng = random.Random(seed)
    food = Position(11, 7)
    groups: dict[
        tuple[float, ...], list[tuple[tuple[Position, ...], frozenset[int]]]
    ] = defaultdict(list)

    for _ in range(attempts):
        blocks = random_connected_body(rng, rng.randint(12, 36), food)
        if blocks is None:
            continue
        actions = optimal_first_actions(blocks, food)
        if not actions:
            continue
        observation = observation_of(blocks, food)
        for previous_body, previous_actions in groups[observation]:
            if previous_actions.isdisjoint(actions):
                return AliasingWitness(
                    observation,
                    previous_body,
                    previous_actions,
                    blocks,
                    actions,
                )
        groups[observation].append((blocks, actions))
    raise RuntimeError(f"在 {attempts} 次采样中未找到混叠见证，请提高 attempts")


def replay_weight_report() -> None:
    rewards = {
        "接近食物": 0.0,
        "距离不变": -0.1,
        "远离食物": -0.2,
        "吃到食物": 10.0,
        "死亡": -10.0,
    }
    weights = {name: abs(value) + 1e-5 for name, value in rewards.items()}
    baseline = weights["接近食物"]
    print("[回放采样权重：单条经验]")
    for name, reward in rewards.items():
        print(
            f"  {name:8s} reward={reward:5.1f}, "
            f"weight={weights[name]:.5f}, 相对接近经验={weights[name] / baseline:,.0f}x"
        )


def verify_potential_shaping(gamma: float = 0.9, eta: float = 0.1) -> None:
    """验证 F=gamma*Phi(s')-Phi(s) 的折扣和只依赖首尾状态。"""
    distances = np.array([9.0, 8.0, 7.0, 8.0, 7.0, 6.0])
    potential = -distances / BOARD_SIZE
    shaping = eta * (gamma * potential[1:] - potential[:-1])
    discounted_sum = sum((gamma**t) * value for t, value in enumerate(shaping))
    boundary_term = eta * (-potential[0] + (gamma ** len(shaping)) * potential[-1])
    np.testing.assert_allclose(discounted_sum, boundary_term, atol=1e-12)
    print("\n[Potential shaping 校验]")
    print(f"  折扣塑形和={discounted_sum:.12f}")
    print(f"  首尾边界项={boundary_term:.12f}（一致）")


def boundary_feature_report() -> None:
    """展示中心点墙距的不对称，以及右边界格未进入射线扫描。"""
    head = Position(7, 7)
    food = Position(10, 10)
    base = observation_of((head, Position(7, 6)), food)
    with_edge_body = observation_of(
        (head, Position(7, 6), Position(13, 7)), food
    )
    print("\n[边界特征校验]")
    print(
        "  中心 (7,7) 编码的左/右/上/下墙距="
        f"{base[22:26]}；对称棋盘应左右、上下一致。"
    )
    print(
        "  在合法边界格 (13,7) 放入身体后，danger_b_r 仍为 "
        f"{int(with_edge_body[8])}，dist_body_r 仍为 {with_edge_body[27]:.1f}。"
    )


def format_actions(actions: frozenset[int]) -> str:
    return "/".join(ACTION_NAMES[action] for action in sorted(actions))


def state_aliasing_report() -> None:
    witness = find_aliasing_witness()
    print("\n[34 维状态混叠见证]")
    print("  两个蛇身的 34 维观测逐项完全相同：", witness.observation)
    print("  棋盘 A 最优首动作：", format_actions(witness.first_actions))
    print("  棋盘 A 蛇身：", list(witness.first_body))
    print("  棋盘 B 最优首动作：", format_actions(witness.second_actions))
    print("  棋盘 B 蛇身：", list(witness.second_body))
    print("  结论：单靠该一维观测无法同时为这两个局面给出正确决策。")


def reset_headless_game(game: Game) -> None:
    """建立不依赖 Pygame 图片、窗口和时钟的等价局面。"""
    from snake_env import Snake

    snake = Snake.__new__(Snake)
    snake.blocks = [Position(7, 7), Position(7, 6)]
    snake.current_direction = Direction.right
    game.snake = snake
    game.berry = SimpleNamespace(position=Position(1, 1))
    game.score = 0
    game.frame = 0
    game.total_step = 0
    game.reward = 0.0
    game.prev_head = snake.blocks[0]
    game.prev_blocks = tuple(snake.blocks)
    game.position_berry()


def evaluate_checkpoint(episodes: int = 30, seed: int = 20260920) -> None:
    """对已有 checkpoint 做小规模贪心回放；不探索、不反传、不保存。"""
    checkpoint_path = ROOT / "model" / "model.pth"
    if not checkpoint_path.exists():
        print("\n[已有模型快速回放]\n  未找到 model/model.pth，跳过。")
        return

    import torch
    from Agent import Linear_QNet

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Linear_QNet(34, 128, 3)
    model.load_state_dict(checkpoint["main_state_dict"])
    model.eval()

    game = make_game((Position(7, 7), Position(7, 6)), Position(1, 1))
    game.path_graph = game.build_path_graph()
    game.draw = lambda: None
    game.Clock = SimpleNamespace(tick=lambda _: None)

    scores: list[int] = []
    action_counts = np.zeros(3, dtype=int)
    timeout_count = 0
    repeated_before_food = 0

    with torch.no_grad():
        for _ in range(episodes):
            reset_headless_game(game)
            seen_before_food: set[tuple[Position, int]] = set()
            repeated = False
            while True:
                marker = (game.snake.blocks[0], game.snake.current_direction)
                if game.score == 0 and marker in seen_before_food:
                    repeated = True
                seen_before_food.add(marker)

                state = torch.as_tensor(game.get_state(), dtype=torch.float32)
                action = int(model(state).argmax().item())
                action_counts[action] += 1
                _, done, score = game.play_step(action)
                if done:
                    if not game.head_hit_body() and not game.head_hit_wall():
                        timeout_count += 1
                    scores.append(score)
                    repeated_before_food += int(repeated and score == 0)
                    break

    total_actions = int(action_counts.sum())
    print("\n[已有模型快速回放：纯贪心、无训练]")
    print(
        f"  checkpoint epoch={checkpoint.get('epoch')}, {episodes} 局平均分="
        f"{np.mean(scores):.3f}, 吃到过食物={sum(score > 0 for score in scores)}/{episodes}"
    )
    print(
        f"  超时结束={timeout_count}/{episodes}, 首次吃食物前出现重复头位姿="
        f"{repeated_before_food}/{episodes}"
    )
    print(
        "  动作比例："
        + ", ".join(
            f"{ACTION_NAMES[action]}={action_counts[action] / total_actions:.1%}"
            for action in range(3)
        )
    )


def main() -> None:
    replay_weight_report()
    verify_potential_shaping()
    boundary_feature_report()
    state_aliasing_report()
    evaluate_checkpoint()
    print("\n诊断完成：未训练、未修改现有代码。")


if __name__ == "__main__":
    main()
