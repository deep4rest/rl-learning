import numpy as np
import random


def monte_carlo_es_epsilon_greedy(
    env,
    max_episodes=1500,
    gamma=0.9,
    episode_length=50,
    epsilon=0.5,
    epsilon_decay=True,
    min_epsilon=0.01,
    epsilon_decay_rate=0.9
):
    """
    Monte Carlo Exploring Starts Control + ε-greedy 版本
        :param env: GridWorld_deterministic 实例
        :param max_episodes: 采样的 episode 数
        :param gamma: 折扣因子 Gamma
        :param episode_length: 每个 episode 的最大步长
        :param epsilon: ε-greedy 策略中的探索概率
        :param epsilon_decay: 是否在训练过程中衰减 ε
        :param min_epsilon: ε 衰减时的下限
        :param epsilon_decay_rate: ε 每次衰减的比例
        :return: (policy, value)
    """

    # 1. 获取环境参数
    rows = env.rows
    columns = env.columns
    n_states = rows * columns
    n_actions = env.actions  # 假设为整数，例如 5

    # 2. 初始化 Q(s,a)、回报累计以及计数
    qtable = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    # 最终要返回的“贪心策略”
    greedy_policy = np.random.randint(0, n_actions, size=n_states)

    # 3. 多次采样 episode
    for episode in range(1, max_episodes + 1):

        # 3.1 Exploring Starts：随机起始 state 和 action
        start_state = random.randint(0, n_states - 1)
        start_action = random.randint(0, n_actions - 1)

        episode_buffer = []   # 存储 (s, a, r)

        state = start_state
        action = start_action

        # 3.2 生成一条 episode
        for t in range(episode_length):

            next_state, reward = env.getScore(state, action)
            episode_buffer.append((state, action, reward))

            state = next_state

            # 后续动作使用 ε-greedy
            if np.random.rand() < epsilon:
                # 探索：随机动作
                action = random.randint(0, n_actions - 1)
            else:
                # 利用：根据当前 Q 选贪心动作
                action = int(np.argmax(qtable[state]))

        # 4. 从 episode 末尾回溯，计算回报并更新 Q(s,a)
        G = 0.0
        for t in reversed(range(len(episode_buffer))):
            s_t, a_t, r_t = episode_buffer[t]
            G = gamma * G + r_t

            # Every-Visit：每次出现 (s_t, a_t) 都更新
            returns_sum[s_t, a_t] += G
            returns_count[s_t, a_t] += 1

            # 增量式平均更新 Q(s,a)
            qtable[s_t, a_t] += (G - qtable[s_t, a_t]) / \
                returns_count[s_t, a_t]

        # 5. 策略提升：对每个状态取 Q 最大的动作，形成贪心策略
        greedy_policy = np.argmax(qtable, axis=1)

        # 6. ε 衰减（可选）
        if epsilon_decay:
            epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)

        # 打印训练进度
        if episode % 500 == 0:
            print(
                f"Episode {episode} finished. epsilon = {epsilon:.4f}")

    # 7. 得到最优状态价值函数 V(s) = max_a Q(s,a)
    value = np.max(qtable, axis=1)

    return greedy_policy, value
