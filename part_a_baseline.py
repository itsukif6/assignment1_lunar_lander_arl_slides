# part_a_baseline.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

def run_random_baseline(n_episodes=100, record=True):
    """執行隨機動作的基線實驗，並統計結果。"""
    if record: # 如果 record=True，則每20回合錄製一段影片
        env = gym.make('LunarLander-v3', render_mode='rgb_array')
        env = RecordVideo(env, 'outputs/part_a/videos/',
                          episode_trigger=lambda x: x % 20 == 0)
    else: # 否則直接創建環境，不錄製影片
        env = gym.make('LunarLander-v3')

    rewards, lengths = [], [] # 用於記錄每回合的總獎勵和步數

    for ep in range(n_episodes):
        obs, _ = env.reset() # 重置環境，獲取初始觀測值 
        total_reward, steps = 0, 0 # 每回合的總獎勵和步數
        done = False # 回合結束標誌
        while not done:
            action = env.action_space.sample()   # 隨機動作
            obs, reward, terminated, truncated, _ = env.step(action) # 執行動作，獲取下一個觀測值、獎勵、是否終止
            total_reward += reward
            steps += 1
            done = terminated or truncated # 回合結束條件：終止或截斷
        rewards.append(total_reward)
        lengths.append(steps)
        print(f"Episode {ep+1:3d} | Reward: {total_reward:7.2f} | Steps: {steps}")

    env.close()

    # 統計
    print(f"\n=== 隨機基線統計 ({n_episodes} 回合) ===")
    print(f"平均獎勵:   {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"最高獎勵:   {np.max(rewards):.2f}")
    print(f"最低獎勵:   {np.min(rewards):.2f}")
    print(f"成功率:     {np.mean(np.array(rewards) >= 200)*100:.1f}%")
    print(f"平均步數:   {np.mean(lengths):.1f}")

    # 繪圖
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, label='Every Episode Reward')
    plt.axhline(np.mean(rewards), color='r', linestyle='--', label=f'Mean={np.mean(rewards):.1f}')
    plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.title('Random Baseline - Rewards'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lengths, alpha=0.6, color='orange')
    plt.xlabel('Episode'); plt.ylabel('Steps')
    plt.title('Random Baseline - Episode Length')
    plt.tight_layout()
    plt.savefig('outputs/part_a/plots/baseline_stats.png', dpi=150)
    plt.show()

    stats = {
        'episode_rewards': rewards,
        'episode_lengths': lengths,
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'success_rate': float(np.mean(np.array(rewards) >= 200))
    }

    return stats

if __name__ == '__main__':
    import os
    os.makedirs('outputs/part_a', exist_ok=True)
    os.makedirs('outputs/part_a/videos', exist_ok=True)
    run_random_baseline()