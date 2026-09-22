import random
from collections import deque
import numpy as np
import torch
# from snake_env import Game
# from Agent import Agent
from helper import plot

from snake_env import Game, Direction, Position
from Agent import Agent, Transition, Linear_QNet

import matplotlib as plt
from pathlib import Path

plt.use("Agg")

def epsilon_by_step(step, start=1.0, end=0.05, decay_steps=30_000):
    ratio = min(step / decay_steps, 1.0)
    return start + ratio * (end - start)


def get_legal_actions(game):
    """过滤下一步必定撞墙或撞身体的动作。"""
    clockwise = [
        Direction.right,
        Direction.down,
        Direction.left,
        Direction.up
    ]

    direction_index = clockwise.index(game.snake.current_direction)
    head = game.snake.blocks[0]
    legal_actions = []

    for action in range(game.nA):
        if action == 0:
            next_index = direction_index
        elif action == 1:
            next_index = (direction_index + 1) % 4
        else:
            next_index = (direction_index - 1) % 4

        next_direction = clockwise[next_index]

        dx, dy = {
            Direction.right: (1, 0),
            Direction.left: (-1, 0),
            Direction.up: (0, -1),
            Direction.down: (0, 1),
        }[next_direction]

        next_head = Position(head.x + dx, head.y + dy)

        if game.head_hit_wall(next_head):
            continue

        # 不吃食物时尾巴会移走，因此旧尾格可以进入。
        if next_head == game.berry.position:
            blocked_body = game.snake.blocks
        else:
            blocked_body = game.snake.blocks[:-1]

        if next_head not in blocked_body:
            legal_actions.append(action)

    # 已经无路可走时保留全部动作，让环境正常返回死亡。
    return legal_actions or list(range(game.nA))


def select_action(agent, game, state, epsilon):
    legal_actions = get_legal_actions(game)

    if random.random() < epsilon:
        # 真正的均匀随机探索
        return random.choice(legal_actions)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        q_values = agent.trainer.model(state_tensor)

        masked_q = torch.full_like(q_values, float("-inf"))
        masked_q[legal_actions] = q_values[legal_actions]

        return int(masked_q.argmax().item())


def train_uniform_memory(agent, batch_size):
    """暂时绕开按 abs(reward) 采样的 ReplayBuffer.sample。"""
    size = min(batch_size, len(agent.memory.memory))

    if size == 0:
        return

    batch = random.sample(list(agent.memory.memory), size)
    transitions = Transition(*zip(*batch))

    states = np.asarray(transitions.state, dtype=np.float32)
    actions = np.asarray(transitions.action, dtype=np.int64)
    rewards = np.asarray(transitions.reward, dtype=np.float32)
    next_states = np.asarray(transitions.next_state, dtype=np.float32)
    dones = np.asarray(transitions.done, dtype=np.float32)

    agent.trainer.train_step(
        states,
        actions,
        rewards,
        next_states,
        dones
    )


def train_v2(
    max_steps=5_000,
    warmup_steps=1_000,
    train_every=4,
    target_update_every=1_000,
    batch_size=256,
    render=False
):
    game = Game()

    agent = Agent(
        game.nS,
        game.nA,
        gamma=0.99,
        lr=1e-3,
        max_memory=50_000,
        hidden_dim=128
    )

    # 使用更稳定的 Huber loss
    agent.trainer.criterion = torch.nn.SmoothL1Loss()

    if not render:
        game.draw = lambda: None
        game.Clock.tick = lambda _: None

    state = game.get_state()
    recent_scores = deque(maxlen=100)

    episode = 0
    total_steps = 0

    while total_steps < max_steps:
        epsilon = epsilon_by_step(total_steps)

        action = select_action(
            agent,
            game,
            state,
            epsilon
        )

        reward, done, score = game.play_step(action)
        next_state = game.get_state()

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        total_steps += 1

        if (
            total_steps >= warmup_steps
            and total_steps % train_every == 0
        ):
            train_uniform_memory(agent, batch_size)

        if total_steps % target_update_every == 0:
            agent.trainer.copy_model()

        state = next_state

        if done:
            episode += 1
            agent.n_game += 1
            recent_scores.append(score)

            game.reset()
            state = game.get_state()

            if episode % 20 == 0:
                print(
                    f"episode={episode}, "
                    f"step={total_steps}, "
                    f"epsilon={epsilon:.3f}, "
                    f"score={score}, "
                    f"mean100={np.mean(recent_scores):.3f}"
                )

                save_dir = Path("./model_v2")
                save_dir.mkdir(parents=True, exist_ok=True)

                torch.save(
                    {
                        "model_state_dict": agent.trainer.model.state_dict(),
                        "target_state_dict": agent.trainer.target_model.state_dict(),
                        "optimizer_state_dict": agent.trainer.optimizer.state_dict(),
                        "steps": total_steps,
                        "episodes": episode,
                        "mean100": float(np.mean(recent_scores)),
                    },
                    save_dir / "model.pth"
                )

    print(
        "短程训练结束：",
        f"episode={episode},",
        f"steps={total_steps},",
        f"mean100={np.mean(recent_scores):.3f}"
    )


def train():
    plot_scores = []
    plot_mean_scores = []
    record = 0
    total_steps = 0
    game = Game()
    agent = Agent(game.nS, game.nA)
    state_new = game.get_state()

    while True:
        state_old = state_new
        final_move = agent.get_action(state_old, agent.n_game)
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()
        agent.remember(state_old, final_move, reward, state_new, done)
        agent.train_long_memory(batch_size=256)
        total_steps += 1

        if total_steps % 20 == 0:
            agent.trainer.copy_model()

        if done:
            game.reset()
            state_new = game.get_state()
            agent.n_game += 1
            if score > record:
                record = score
                agent.trainer.save(best_reward=record, epoch=agent.n_game)
            print('Game', agent.n_game, 'Score', score, 'Record', record)
            plot_scores.append(score)
            mean_scores = np.mean(plot_scores[-10:])
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores)

        # 使用整局得分判断目标，避免吃到一次食物（reward=10）就停止训练。
        if done and score >= 3:
            agent.trainer.save(best_reward=record, epoch=agent.n_game)
            break

def play():
    record = 0
    total_steps = 0
    game = Game()
    agent = Agent(game.nS, game.nA)
    state_new = game.get_state()

    while True:
        state_old = state_new
        final_move = agent.get_action(state_old, agent.n_game)
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()
        total_steps += 1

        if done:
            game.reset()
            state_new = game.get_state()
            agent.n_game += 1
            if score > record:
                record = score
            print('Game', agent.n_game, 'Score', score, 'Record', record)

def select_play_action(model, game, state):
    legal_actions = get_legal_actions(game)

    with torch.no_grad():
        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32
        )

        q_values = model(state_tensor)

        # 保留与训练时一致的必死动作掩码
        masked_q = torch.full_like(
            q_values,
            float("-inf")
        )
        masked_q[legal_actions] = q_values[legal_actions]

        return int(masked_q.argmax().item())


def play_v2(episodes=10):
    game = Game()

    model = Linear_QNet(
        input_size=game.nS,
        hidden_size=128,
        output_size=game.nA
    )

    checkpoint = torch.load(
        "./model_v2/model.pth",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    state = game.get_state()
    episode = 0

    while episode < episodes:
        # 纯贪心运行，不进行随机探索
        action = select_play_action(
            model,
            game,
            state
        )

        _, done, score = game.play_step(action)
        state = game.get_state()

        if done:
            episode += 1

            print(
                f"Play episode={episode}, "
                f"score={score}"
            )

            game.reset()
            state = game.get_state()


if __name__ == '__main__':
    # train()
    # play()
    train_v2(max_steps=5_000, render=False)
    # play_v2(episodes=10)
