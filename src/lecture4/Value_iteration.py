import numpy as np


def value_iteration(env, gamma=0.9, threshold=1e-6, max_iterations=200):
    """
    Value Iteration Algorithm.
        :param env: 环境对象
        :param gamma: 折扣因子
        :param threshold: 收敛阈值
        :param max_iterations: 最大迭代次数
        :return: (policy, value)
    """
    # 1. 动态获取环境参数
    rows = env.rows
    columns = env.columns
    n_states = rows * columns
    n_actions = env.actions  # 假设动作空间为 5

    # 2. 初始化 Value 和 Q-table
    value = np.zeros(n_states)
    qtable = np.zeros((n_states, n_actions))

    iteration = 0
    while True:

        pre_value = np.copy(value)
        for state in range(rows*columns):
            for action in range(n_actions):
                next_state, reward = env.getScore(state, action)
                qtable[state][action] = reward + gamma * pre_value[next_state]

        # 3. 更新 Policy 和 Value
        policy = np.argmax(qtable, axis=1)  # policy update
        value = np.max(qtable, axis=1)  # value update

        # 4. 检查收敛条件
        diff = np.sum((value - pre_value) ** 2)
        iteration += 1

        # 5. 打印当前迭代信息
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Diff = {diff:.9f}")
        if iteration > max_iterations or diff < threshold:
            break

    # print(value)
    # env.showPolicy(policy)

    return policy, value
