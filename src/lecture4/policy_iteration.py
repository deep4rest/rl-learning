import numpy as np
# import env.Gridworld_deterministic as gwd


def policy_iteration(env,
                     gamma=0.9,
                     threshold=1e-6,
                     max_iterations=200,
                     max_policy_evaluations=50
                     ):
    """
    Policy Iteration Algorithm.
        :param env: 传入的环境对象 (GridWorld)
        :param gamma: 折扣因子
        :param threshold: 策略评估的收敛阈值
        :param max_iterations: 最大策略迭代次数
        :param max_policy_evaluations: 每次迭代中策略评估的最大循环次数
        :return: 最终的 policy 和 value
    """

    # 从环境对象中动态获取行列数
    rows = env.rows
    columns = env.columns
    n_states = rows * columns
    n_actions = env.actions  # 假设有5个动作

    # 初始化
    value = np.zeros(n_states)
    qtable = np.zeros((n_states, n_actions))

    # 随机初始化策略
    policy = np.random.randint(0, n_actions, size=n_states)

    iteration = 0
    while True:

        for k in range(max_policy_evaluations):  # Policy Evaluation
            pre_value = np.copy(value)
            offset = 0

            for state in range(n_states):
                next_state, reward = env.getScore(state, policy[state])
                value[state] = reward + gamma * pre_value[next_state]
                offset += (value[state] - pre_value[state]) ** 2
            if offset < threshold:  # 判断策略评估是否收敛
                break

        for state in range(n_states):
            for action in range(n_actions):
                next_state, reward = env.getScore(state, action)
                qtable[state][action] = reward + gamma * value[next_state]

        policy_stable = False  # 判断策略提升是否收敛
        new_policy = np.argmax(qtable, axis=1)  # policy update
        if np.array_equal(new_policy, policy):
            policy_stable = True
        policy = new_policy

        iteration += 1
        if iteration % 10 == 0:
            print(f"Iteration {iteration} done. loss={offset:.9f}")

        if policy_stable:
            print(f"Algorithm converged at iteration {iteration}")
            break
        if iteration >= max_iterations:
            print(f"Stopped at max_iterations = {max_iterations}")
            break

    # print("Final Value Function:")
    # print(value.reshape(rows, columns))
    # print("\nFinal Policy:")
    # env.showPolicy(policy)

    return policy, value
