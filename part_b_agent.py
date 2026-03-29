# part_b_agent.py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ── 1. 經驗回放緩衝區 ─────────────────────────────────
class ReplayBuffer:
    """簡單的經驗回放緩衝區，使用 deque 實現固定容量的 FIFO 緩衝區。"""
    def __init__(self, capacity=100_000): # 初始化緩衝區，設定最大容量
        self.buffer = deque(maxlen=capacity) # 使用 deque 來自動丟棄最舊的經驗

    def push(self, state, action, reward, next_state, done): # 將經驗加入緩衝區
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): # 從緩衝區中隨機抽取一批經驗
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch) # 解壓批次中的經驗，分別獲取狀態、動作、獎勵、下一狀態和是否終止
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buffer)

# ── 2. Q 網路 ──────────────────────────────────────────
class QNetwork(nn.Module):
    """簡單的全連接神經網路，輸入為狀態，輸出為每個動作的 Q 值。"""
    def __init__(self, state_dim=8, action_dim=4, hidden=256): # 初始化 Q 網路，設定狀態維度、動作維度和隱藏層大小
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x): # 前向傳播，輸入狀態，輸出動作的 Q 值
        return self.net(x)

# ── 3. DQN Agent ───────────────────────────────────────
class DQNAgent:
    """DQN Agent 實現，包含 ε-greedy 策略、經驗回放和目標網路更新等功能。"""
    def __init__(self, state_dim=8, action_dim=4,
                 lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=10,
                 buffer_size=100_000): # 初始化 DQN Agent，設定各種超參數

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 線上網路用於選擇動作和更新，目標網路用於計算 TD 目標，定期從線上網路同步參數
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.MSELoss()

    # ε-greedy 選動作
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon: # 訓練模式下以 ε 的概率選擇隨機動作
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) # 將狀態轉換為張量，並添加批次維度
        with torch.no_grad(): # 評估模式下直接選擇 Q 值最高的動作
            return self.q_net(state_t).argmax(dim=1).item()

    # 單步更新
    def update(self):
        if len(self.replay_buffer) < self.batch_size: # 如果緩衝區中的經驗不足以抽取一批，則不進行更新
            return None

        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size) # 從緩衝區中抽取一批經驗
        s, a, r, ns, d = s.to(self.device), a.to(self.device), \
                         r.to(self.device), ns.to(self.device), d.to(self.device) # 將抽取的經驗轉移到設備上（CPU 或 GPU）

        # 計算 TD 目標
        with torch.no_grad():
            next_q = self.target_net(ns).max(dim=1)[0] # 目標網路計算下一狀態的最大 Q 值
            target = r + self.gamma * next_q * (1 - d) # TD 目標：獎勵 + 折扣後的下一狀態最大 Q 值（如果未終止）

        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1) # 線上網路計算當前狀態下選擇動作的 Q 值
        loss = self.loss_fn(current_q, target) # 計算損失（均方誤差）

        self.optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向傳播計算梯度
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0) # 梯度裁剪，防止梯度爆炸
        self.optimizer.step() # 更新線上網路的參數

        # 衰減 ε
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 定期更新目標網路
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0: # 每 target_update_freq 步更新一次目標網路
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path='model.pth'):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path='model.pth'):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))