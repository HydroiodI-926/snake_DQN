import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
import os

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, alpha=1.0):
        '''
        由于在训练初期蛇吃到食物的概率过低，吃到食物的经验往往需要更频繁的复习
        因此设置一个权重记忆缓冲池
        :param capacity: 缓存池的容量
        :param alpha: 权重
        '''
        self.memory = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        rewards = np.array([abs(m.reward) for m in self.memory], dtype=np.float32)

        probs = (rewards + 1e-5) ** self.alpha  # 避免在一段时间中始终没有得分导致概率为NaN
        probs /= probs.sum()  # 归一化

        if len(self.memory) > batch_size:
            indices = np.random.choice(len(self.memory), batch_size, replace=False, p=probs)
        else:
            indices = np.random.choice(len(self.memory), len(self.memory), p=probs)

        batch = [self.memory[idx] for idx in indices]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # 前向传播
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, lr, gamma, input_dim, hidden_dim, output_dim, n_game=0, best_reward=0):
        self.gamma = gamma
        self.hidden_size = hidden_dim
        self.n_game = n_game
        self.best_reward = best_reward
        self.model = Linear_QNet(input_dim, self.hidden_size, output_dim)
        self.target_model = Linear_QNet(input_dim, self.hidden_size, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load()
        self.criterion = nn.MSELoss()
        self.copy_model()


    def load(self, model_name='model.pth'):
        model_folder_path = './model'
        try:
            model_name = os.path.join(model_folder_path, model_name)
            checkpoint = torch.load(model_name)

            self.model.load_state_dict(checkpoint['main_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.n_game = checkpoint['epoch']
            self.best_reward = checkpoint['best_reward']

            print("模型导入成功")
            print("-----------------------")
            print(f"上一次训练到轮次{self.n_game}\n最好成绩是{self.best_reward}")
            print("-----------------------")

        except Exception as e:
            print("模型导入失败(将创建新的网络进行训练)：\n", e)

    def copy_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, model_name='model.pth', best_reward=0, epoch=0):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_name = os.path.join(model_folder_path, model_name)

        torch.save({
            'main_state_dict': self.model.state_dict(),
            'target_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_reward': best_reward,
            'epoch': epoch
        }, model_name)

        print("保存成功")


    def train_step(self, state, action, reward, next_state, done):

        # 将numpy数组转化为tensor张量
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        action = torch.unsqueeze(action, -1)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        Q_value = self.model(state).gather(-1, action).squeeze()
        Q_value_next = self.target_model(next_state).detach().max(-1)[0]
        target = (reward + self.gamma * Q_value_next * (1 - done)).squeeze()

        self.optimizer.zero_grad()
        loss = self.criterion(Q_value, target)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self, nS, nA, max_explore=100, gamma=0.9,
                 lr=0.001, max_memory=5000, hidden_dim=128, best_reward=0):
        self.max_explore = max_explore
        self.memory = ReplayBuffer(max_memory, 1.0)
        self.nS = nS
        self.nA = nA
        self.trainer = QTrainer(lr, gamma, self.nS, hidden_dim, self.nA)
        self.n_game = self.trainer.n_game
        self.best_reward = self.trainer.best_reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_long_memory(self, batch_size):

        trans = self.memory.sample(batch_size)

        # 转tensor
        states = torch.from_numpy(np.array(trans.state, dtype=np.float32))
        actions = torch.from_numpy(np.array(trans.action, dtype=np.float32))
        rewards = torch.from_numpy(np.array(trans.reward, dtype=np.float32))
        next_states = torch.from_numpy(np.array(trans.next_state, dtype=np.float32))
        dones = torch.from_numpy(np.array(trans.done, dtype=np.float32))

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, n_game, explore=True):
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.trainer.model(state).detach().squeeze()
        epsilon = self.max_explore - n_game
        if explore and random.randint(0, self.max_explore) < epsilon:
            # logits = prediction - np.max(prediction)
            # prob = np.exp(logits) / np.exp(logits).sum()  容易梯度爆炸
            prob = torch.softmax(prediction, dim=0).cpu().numpy()

            prob = prob + 1e-8
            prob /= prob.sum()  # 平滑化，归一化

            final_move = np.random.choice(len(prob), p=prob)
        else:
            final_move = prediction.argmax().item()



















        return final_move
