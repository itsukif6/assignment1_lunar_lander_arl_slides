import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os

# Hyperparameters
LEARNING_RATE = 5e-4
GAMMA = 0.99 # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 10 # Update target network every N episodes

# Create environment
env = gym.make('LunarLander-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
env.close()

print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")

import os
for path in ['outputs/part_a/plots', 
             'outputs/part_c/plots', 'outputs/part_c/checkpoints', 
             'outputs/part_c/videos/training', 'outputs/part_c/videos/trained',
             'outputs/part_d/plots/']:
    os.makedirs(path, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Part A — 環境設置與隨機基線
#   - 建立 LunarLander-v3 環境
#   - 執行 100 回合隨機策略
#   - 收集平均獎勵、回合長度、成功率
#   - 儲存 3~5 段 videos
# ═══════════════════════════════════════════════════════════════════════════════
from part_a_baseline import run_random_baseline
from utils import print_stats, plot_baseline, record_episodes

print("\n" + "="*60)
print("Part A: Environment Setup & Random Baseline")
print("="*60)

# 執行隨機基線（100 回合），回傳 stats dict
baseline_stats = run_random_baseline(n_episodes=100, record=True)

# 使用 utils 提供的格式化輸出函式顯示摘要
print_stats(baseline_stats)

# 使用 utils 提供的繪圖函式儲存 2×2 統計圖
plot_baseline(baseline_stats, out_path='outputs/part_a/plots/baseline_stats.png')

# ═══════════════════════════════════════════════════════════════════════════════
# Part B — Agent 實作
#   - 神經網路 Q 函數近似器（QNetwork）
#   - 經驗回放緩衝區（ReplayBuffer）
#   - 目標網路定期同步（DQNAgent.target_net）
#   - ε-greedy 探索策略
#   - 至少 500 回合的訓練迴圈
#
# 實作位於 part_b_agent.py，以下匯入驗證三個類別可正常初始化。
# ═══════════════════════════════════════════════════════════════════════════════
from part_b_agent import ReplayBuffer, QNetwork, DQNAgent

print("\n" + "="*60)
print("Part B: Agent Implementation — Component Check")
print("="*60)

# 驗證 ReplayBuffer 可正常存取
_buf = ReplayBuffer(capacity=1000)
_dummy_s = np.zeros(state_dim)
_buf.push(_dummy_s, 0, 0.0, _dummy_s, False)
print(f"ReplayBuffer OK  | size={len(_buf)}")

# 驗證 QNetwork 前向傳播
_qnet = QNetwork(state_dim=state_dim, action_dim=action_dim)
_x = torch.zeros(1, state_dim)
_out = _qnet(_x)
print(f"QNetwork OK      | output shape={tuple(_out.shape)}")

# 驗證 DQNAgent 可初始化
_agent_check = DQNAgent(state_dim=state_dim, action_dim=action_dim)
print(f"DQNAgent OK      | device={_agent_check.device}")

# ═══════════════════════════════════════════════════════════════════════════════
# Part C — 訓練與分析
#   - 追蹤每回合獎勵、損失、ε 衰減、平均 Q 值
#   - 繪製含移動平均的學習曲線
#   - 以 100 回合無探索評估訓練後的策略
#   - 錄製 3~5 段已訓練 Agent 的 GIF 影片
# ═══════════════════════════════════════════════════════════════════════════════
from part_c_train_eval import train, evaluate
from utils import plot_training_curves, save_checkpoint, load_checkpoint

print("\n" + "="*60)
print("Part C: Training & Analysis")
print("="*60)

# 訓練（最多 600 回合，達標提前停止）
trained_agent, metrics = train(n_episodes=600)

# 使用 utils 提供的繪圖函式儲存訓練曲線（含 solved_at 標記線）
plot_training_curves(metrics, out_dir='outputs/part_c/plots')

# 使用 utils 儲存最終 checkpoint
save_checkpoint(
    trained_agent,
    episode=len(metrics['episode_rewards']),
    rewards=metrics['episode_rewards'],
    filename='outputs/part_c/checkpoints/model_final.pt'
)

# 評估（100 回合，關閉探索）並統計
eval_rewards = evaluate(trained_agent, n_episodes=100, record=True)

# 印出評估摘要（套用 utils.print_stats 格式）
eval_stats = {
    'mean_reward':  float(np.mean(eval_rewards)),
    'std_reward':   float(np.std(eval_rewards)),
    'min_reward':   float(np.min(eval_rewards)),
    'max_reward':   float(np.max(eval_rewards)),
    'mean_length':  0.0,  # evaluate() 未追蹤步數，填 0 作佔位
    'success_rate': float(np.mean(np.array(eval_rewards) >= 200)),
}
print("\n--- Evaluation Stats ---")
print_stats(eval_stats)

# ═══════════════════════════════════════════════════════════════════════════════
# Part D — 超參數實驗與報告
#   - 測試至少 3 種超參數變體：學習率、ε 衰減、目標網路更新頻率
#   - 繪製各組訓練曲線比較圖
#   - 實驗結果供 2~3 頁書面報告使用
# ═══════════════════════════════════════════════════════════════════════════════
from part_d_experiments import run_experiment

print("\n" + "="*60)
print("Part D: Hyperparameter Experiments")
print("="*60)

# 執行全部三組實驗（學習率 / ε 衰減速度 / 目標網路更新頻率）
# 比較圖自動儲存至 part_d/plots/hyperparameter_comparison.png
from part_d_experiments import configs_lr, configs_eps, configs_tuf, run_experiment
from part_d_experiments import moving_avg

# 執行實驗並收集結果
results_lr = {label: run_experiment(config, label=label) for config, label in configs_lr}
results_eps = {label: run_experiment(config, label=label) for config, label in configs_eps}
results_tuf = {label: run_experiment(config, label=label) for config, label in configs_tuf}

# 畫比較圖
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, results, title in zip(axes,
        [results_lr, results_eps, results_tuf],
        ['Learning Rate Comparison', 'ε Decay Speed Comparison', 'Target Update Frequency Comparison']):
    for label, r in results.items():
        ax.plot(moving_avg(r), label=label)
    ax.axhline(200, color='r', linestyle='--', alpha=0.5, label='Target=200')
    ax.set_title(title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward (20)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('outputs/part_d/plots/hyperparameter_comparison.png', dpi=150)
plt.show()

# 最後總結輸出路徑
print("\nAll parts completed.")
print("Outputs:")
print("  Part A plots   → outputs/part_a/plots/")
print("  Part A videos  → outputs/part_a/videos/")
print("  Part C plots   → outputs/part_c/plots/")
print("  Part C ckpt    → outputs/part_c/checkpoints/")
print("  Part C videos  → outputs/part_c/videos/")
print("  Part D plots   → outputs/part_d/plots/")
