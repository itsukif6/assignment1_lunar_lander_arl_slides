# part_d_experiments.py
import numpy as np, matplotlib.pyplot as plt, gymnasium as gym, os
from part_b_agent import DQNAgent

os.makedirs('part_d_plots', exist_ok=True)

 # 執行一個實驗，根據給定的配置訓練 DQN Agent，並返回每回合的獎勵
def run_experiment(config, n_episodes=400, label=''):
    env = gym.make('LunarLander-v3')
    agent = DQNAgent(**config)
    rewards = []
    
    # 每回合內持續互動直到終止，並記錄獎勵
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0, False
        
        # 每回合內持續互動直到終止
        while not done:
            action = agent.select_action(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.replay_buffer.push(obs, action, r, next_obs, float(done))
            agent.update()
            total_reward += r
            obs = next_obs
            
        rewards.append(total_reward)
        avg = np.mean(rewards[-100:])
        print(f"[{label}] Ep {ep+1:3d} | Avg100={avg:.1f}", end='\r')
        
    env.close()
    print()
    return rewards

def moving_avg(data, w=20):
    return [np.mean(data[max(0,i-w):i+1]) for i in range(len(data))]

# ── 實驗1：不同學習率 ─────────────────────────
print("=== 實驗1：Learning Rate ===")
configs_lr = [
    ({'lr': 1e-4}, 'lr=1e-4 (slow)'),
    ({'lr': 1e-3}, 'lr=1e-3 (default)'),
    ({'lr': 5e-3}, 'lr=5e-3 (fast)'),
]
results_lr = {label: run_experiment(cfg, label=label)
              for cfg, label in configs_lr}

# ── 實驗2：不同 ε 衰減速度 ───────────────────
print("=== 實驗2：Epsilon Decay ===")
configs_eps = [
    ({'epsilon_decay': 0.990}, 'decay=0.990 (fast)'),
    ({'epsilon_decay': 0.995}, 'decay=0.995 (default)'),
    ({'epsilon_decay': 0.999}, 'decay=0.999 (slow)'),
]
results_eps = {label: run_experiment(cfg, label=label)
               for cfg, label in configs_eps}

# ── 實驗3：目標網路更新頻率 ──────────────────
print("=== 實驗3：Target Update Frequency ===")
configs_tuf = [
    ({'target_update_freq': 5},  'freq=5  (frequent)'),
    ({'target_update_freq': 10}, 'freq=10 (default)'),
    ({'target_update_freq': 50}, 'freq=50 (infrequent)'),
]
results_tuf = {label: run_experiment(cfg, label=label)
               for cfg, label in configs_tuf}

# 畫比較圖
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 每個子圖比較不同配置的訓練曲線，並標註目標獎勵線
for ax, results, title in zip(axes,
        [results_lr, results_eps, results_tuf],
        ['Learning Rate Comparison', 'ε Decay Speed Comparison', 'Target Update Frequency Comparison']):
    for label, r in results.items():
        ax.plot(moving_avg(r), label=label)
    ax.axhline(200, color='r', linestyle='--', alpha=0.5, label='Target=200')
    ax.set_title(title); ax.set_xlabel('Episode'); ax.set_ylabel('Avg Reward (20)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('part_d_plots/hyperparameter_comparison.png', dpi=150)
plt.show()