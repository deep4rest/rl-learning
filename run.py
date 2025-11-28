import numpy as np
from src.env.Gridworld_deterministic import GridWorld_deterministic
from src.lecture4.policy_iteration import policy_iteration
from src.lecture4.Value_iteration import value_iteration
from src.lecture5.mc_basic import monte_carlo_basic

if __name__ == "__main__":

    # 1. 配置并实例化环境
    env = GridWorld_deterministic(
        rows=5,
        columns=5,
        forbiddenAreaNums=7,
        targetNums=1,
        # seed=42,
        forbiddenAreaScore=-1,
        targetscore=1,
        design=[
            "00000",
            "0###0",
            "0#*#0",
            "0#0#0",
            "00000",
        ]
    )

    print("\n====== 1. 初始地图状态 ======")
    env.show()  # 打印空地图

    # 2. 调用算法函数
    print("\n====== 2. 算法运行 ======")
    final_policy, final_value = monte_carlo_basic(
        env,
        gamma=0.5,
        episode_length=50,
    )

    # 3. 展示最终结果
    print("\n====== 3. 最终结果展示 ======")

    print("\n[Final Value Function]:")

    formatted_value = final_value.reshape(env.rows, env.columns)
    print(formatted_value)

    print("\n[Final Policy Map]:")
    # 调用环境的 showPolicy 来展示最终路线
    env.showPolicy(final_policy)
