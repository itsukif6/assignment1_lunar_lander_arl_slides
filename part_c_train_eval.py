# part_c_train_eval.py
import os, gymnasium as gym, numpy as np, matplotlib.pyplot as plt, torch
from gymnasium.wrappers import RecordVideo
from part_b_agent import DQNAgent

 # 訓練 DQN Agent，並在訓練過程中記錄獎勵、損失、ε 衰減和 Q 值變化
def train(n_episodes=600):
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env = RecordVideo(env, 'outputs/part_c/videos/training',
                      episode_trigger=lambda x: x % 50 == 0)

    agent = DQNAgent()
    rewards_log, loss_log, eps_log, q_log = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, ep_losses, ep_qs = 0, [], []
        done = False

        # 每回合內持續互動直到終止
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(obs, action, reward, next_obs, float(done))

            loss = agent.update()
            # 記錄損失（如果有的話）
            if loss is not None:
                ep_losses.append(loss)

            # 記錄平均 Q 值
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
                q_val = agent.q_net(st).max().item()
                ep_qs.append(q_val)

            total_reward += reward
            obs = next_obs

        # 每回合結束後記錄獎勵、損失、ε 和 Q 值
        rewards_log.append(total_reward)
        loss_log.append(np.mean(ep_losses) if ep_losses else 0)
        eps_log.append(agent.epsilon)
        q_log.append(np.mean(ep_qs) if ep_qs else 0)

        # 每 10 回合打印一次訓練進度
        avg100 = np.mean(rewards_log[-100:])
        print(f"Ep {ep+1:4d} | R={total_reward:7.2f} | Avg100={avg100:7.2f} | ε={agent.epsilon:.3f}")

        if (ep + 1) % 50 == 0:
            agent.save(f'outputs/part_c/checkpoints/model_ep{ep+1}.pth')

        solved_at = None  # 預設沒解決
        if avg100 >= 200 and ep >= 99:
            solved_at = ep + 1
            print(f"\n已解決：episode {solved_at}")
            break

    env.close()
    agent.save('outputs/part_c/checkpoints/model_final.pth')

    metrics = {
        'episode_rewards': rewards_log,
        'avg_losses': loss_log,
        'epsilons': eps_log,
        'mean_q_values': q_log,
        'solved_at': solved_at
    }

    plot_training(metrics)

    return agent, metrics

# 繪製訓練過程中的獎勵、損失、ε 衰減和 Q 值變化
def plot_training(metrics):
    rewards  = metrics['episode_rewards']
    losses   = metrics['avg_losses']
    epsilons = metrics['epsilons']
    q_vals   = metrics['mean_q_values']

    def moving_avg(data, w=20):
        return [np.mean(data[max(0,i-w):i+1]) for i in range(len(data))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0,0].plot(rewards, alpha=0.3, label='Every Episode')
    axes[0,0].plot(moving_avg(rewards), label='Moving Average (20)', linewidth=2)
    axes[0,0].axhline(200, color='r', linestyle='--', label='Target=200')
    axes[0,0].set_title('Episode Rewards'); axes[0,0].legend()

    axes[0,1].plot(losses)
    axes[0,1].set_title('Training Loss')

    axes[1,0].plot(epsilons, color='orange')
    axes[1,0].set_title('Epsilon Decay')

    axes[1,1].plot(q_vals, color='green')
    axes[1,1].set_title('Average Q-values')

    plt.tight_layout()
    plt.savefig('outputs/part_c/plots/training_curves.png', dpi=150)
    plt.show()

 # 評估訓練好的模型，並統計結果
def evaluate(agent, n_episodes=100, record=True):
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    if record:
        env = RecordVideo(env, 'outputs/part_c/videos/trained/',
                          episode_trigger=lambda x: x < 5)  # 前5回錄影

    agent.epsilon = 0.0   # 關閉探索
    rewards = []

    # 每回合內持續互動直到終止，並記錄獎勵
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0, False
        while not done:
            action = agent.select_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    env.close()
    print(f"\n=== 測試結果 ({n_episodes} 回合, 無探索) ===")
    print(f"平均獎勵: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"成功率:   {np.mean(np.array(rewards) >= 200)*100:.1f}%")
    print(f"最高/最低: {np.max(rewards):.1f} / {np.min(rewards):.1f}")
    return rewards

if __name__ == '__main__':
    os.makedirs('outputs/part_c', exist_ok=True)
    os.makedirs('outputs/part_c/plots', exist_ok=True)
    os.makedirs('outputs/part_c/checkpoints', exist_ok=True)
    os.makedirs('outputs/part_c/videos', exist_ok=True)
    os.makedirs('outputs/part_c/videos/training', exist_ok=True)
    os.makedirs('outputs/part_c/videos/trained', exist_ok=True)
    agent, train_rewards = train(n_episodes=600)
    eval_rewards = evaluate(agent, n_episodes=100)