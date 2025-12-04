import numpy as np
import random
from collections import defaultdict


def monte_carlo_exploring_starts(env, max_episodes=50,  gamma=0.9, episode_length=50):
    """
    Monte Carlo Exploring Starts Control, Every-Visit 版本
        :param env: GridWorld_deterministic 实例
        :param max_episodes: 采样的 episode 数
        :param gamma: 折扣因子
        :param episode_length: 每个 episode 的最大步长
        :return: (policy, value)
    """

    # 1. 动态获取环境参数
    rows = env.rows
    columns = env.columns
    n_states = rows * columns
    n_actions = env.actions   # 假设动作空间为 n_actions

    # 2. 初始化 Q(s,a)、Policy、以及累计回报统计
    qtable = np.zeros((n_states, n_actions))
    policy = np.random.randint(0, n_actions, size=n_states)

    # 用于 Every-Visit MC 的累计和计数（按平均值估计）
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    for episode in range(1, max_episodes + 1):

        # 3. Exploring Starts: 随机选择起始 state 和 action
        start_state = random.randint(0, n_states - 1)
        start_action = random.randint(0, n_actions - 1)

        episode_buffer = []  # 存储 (state, action, reward)

        # 4. 生成一个 episode
        state = start_state
        action = start_action
        for t in range(episode_length):
            next_state, reward = env.getScore(state, action)
            episode_buffer.append((state, action, reward))

            # 到达下一个 state 后，按当前 policy 选 action
            state = next_state
            action = policy[state]

        # 5. Every-Visit Monte Carlo 更新 Q(s,a)
        # 使用一条 episode 中，从该时间步往后的 return G 来估计该 (s,a) 的 action value
        G = 0.0
        # 逆序遍历 episode，以便从后往前累积折扣回报
        for t in reversed(range(len(episode_buffer))):
            s_t, a_t, r_t = episode_buffer[t]
            G = gamma * G + r_t  # 从该时间步往后的 return

            # Every-Visit：不做是否第一次出现的判断，每次出现都更新
            returns_sum[s_t, a_t] += G
            returns_count[s_t, a_t] += 1

            # 增量式平均（等价于 returns_sum / returns_count）
            qtable[s_t, a_t] += (G - qtable[s_t, a_t]) / \
                returns_count[s_t, a_t]

        # 6. Policy Improvement：对每个 state 选 Q 最大的动作
        policy = np.argmax(qtable, axis=1)

        if episode % 500 == 0:
            print(f"Episode {episode} done.")

    # 7. 从 Q(s,a) 得到状态价值函数 V(s) = max_a Q(s,a)
    value = np.max(qtable, axis=1)

    return policy, value
