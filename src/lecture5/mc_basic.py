import numpy as np
import random
from collections import defaultdict
# import env.Gridworld_deterministic as gwd


def monte_carlo_basic(env, max_iterations=200, gamma=0.9, episode_length=50):
    """
    Monte Carlo Basic Algorithm.
        :param env: GridWorld_deterministic 实例
        :param max_iterations: 迭代次数
        :param gamma: 折扣因子 Gamma
        :param episode_length: 防止死循环的最大步数限制
        :return: (policy,value)
    """

    # 1. 动态获取环境参数
    rows = env.rows
    columns = env.columns
    n_states = rows * columns
    n_actions = env.actions  # 假设动作空间为 5

    # 2. 初始化
    value = np.zeros(n_states)
    qtable = np.zeros((n_states, n_actions))
    policy = np.random.randint(0, n_actions, size=n_states)

    iteration = 0
    while True:

        for state in range(n_states):
            for action in range(n_actions):
                now_state = state
                total_return = 0
                for step in range(episode_length):
                    now_action = action if step == 0 else policy[now_state]
                    next_state, reward = env.getScore(now_state, now_action)
                    total_return += (gamma ** step) * reward
                    now_state = next_state
                qtable[state][action] = total_return

        # 3. 更新 Policy 和 Value
        policy_stable = False  # 判断策略提升是否收敛
        new_policy = np.argmax(qtable, axis=1)  # policy update
        if np.array_equal(new_policy, policy):
            policy_stable = True
        policy = new_policy

        value = np.max(qtable, axis=1)  # value update
        iteration += 1
        if iteration % 10 == 0:
            print(f"Iteration {iteration} done.")
        if policy_stable:
            print(f"Algorithm converged at iteration {iteration}")
            break
        if iteration >= max_iterations:
            print(f"Algorithm stopped at max iteration {iteration}")
            break

    return policy, value
