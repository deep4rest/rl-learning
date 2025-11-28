import numpy as np
import random


class GridWorld_deterministic():
    DIRECTION = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
    ACTION_MAP = {0: '⏫', 1: '⏩', 2: '⏬', 3: '⏪', 4: '🔄'}
    EMOJI = {0: "⬜️"}

    def __init__(
            self,
            rows=5,
            columns=5,
            seed=42,
            forbiddenAreaNums=4,
            forbiddenAreaScore=-1,
            targetNums=1,
            targetscore=1,
            design=None
    ):

        self.rows = rows
        self.columns = columns
        self.seed = seed

        self.actions = 5  # 上、右、下、左、停留原地

        self.forbiddenAreaNums = forbiddenAreaNums
        self.forbiddenAreaScore = forbiddenAreaScore
        self.targetNums = targetNums
        self.targetscore = targetscore

        self.EMOJI[self.forbiddenAreaScore] = "❌"
        self.EMOJI[self.targetscore] = "✅"

        if design is not None:
            self._load_design(design)
        else:
            self._random_generate()

    def _load_design(self, design):
        self.rows = len(design)
        self.columns = len(design[0])

        grid = []
        for row in design:
            grid.append([
                self.forbiddenAreaScore if ch == '#'
                else self.targetscore if ch == '*'
                else 0
                for ch in row
            ])

        self.scoreMap = np.array(grid)
        self.stateMap = [
            [i * self.columns + j for j in range(self.columns)]
            for i in range(self.rows)]
        return

    def _random_generate(self):
        random.seed(self.seed)
        n = self.rows * self.columns
        l = [i for i in range(n)]
        random.shuffle(l)

        g = np.zeros(n, dtype=int)
        for i in l[:self.forbiddenAreaNums]:
            g[i] = self.forbiddenAreaScore
        for i in l[-self.targetNums:]:
            g[i] = self.targetscore

        self.scoreMap = np.array(g).reshape(self.rows, self.columns)
        self.stateMap = [
            [i * self.columns + j for j in range(self.columns)]
            for i in range(self.rows)]

    def show(self):

        for i in range(self.rows):
            print("".join(self.EMOJI[v] for v in self.scoreMap[i]))

    def getScore(self, nowState, action):
        nowx = nowState // self.columns
        nowy = nowState % self.columns

        if (nowx < 0 or nowy < 0 or nowx >= self.rows or nowy >= self.columns):
            print(f"coordinate error: ({nowx},{nowy})")
        if (action < 0 or action >= 5):
            print(f"action error: ({action})")

        nextx = nowx + self.DIRECTION[action][0]
        nexty = nowy + self.DIRECTION[action][1]

        if nextx < 0 or nexty < 0 or nextx >= self.rows or nexty >= self.columns:
            return nowState, -1
        else:
            nextState = nextx * self.columns + nexty
            reward = self.scoreMap[nextx][nexty]
            return nextState, reward

    def showPolicy(self, policy):

        for i in range(self.rows):
            s = ""
            for j in range(self.columns):
                state = i * self.columns + j
                s = s + self.ACTION_MAP[policy[state]]
            print(s)


#  def _find_terminal_states(self):
#         terminals = np.where(self.score_map == self.target_score)
#         states = [(x * self.columns + y)
#                   for x, y in zip(terminals[0], terminals[1])]
#         return set(states)

#     # ----------------------------------------------------------------------
#     # Gym API：reset()
#     # ----------------------------------------------------------------------
#     def reset(self):
#         """重置 agent 到非 forbidden、非 target 的随机位置。"""
#         valid_positions = np.where(self.score_map == 0)
#         idx = random.choice(range(len(valid_positions[0])))
#         x, y = valid_positions[0][idx], valid_positions[1][idx]
#         self.agent_state = x * self.columns + y
#         return self.agent_state

#     # ----------------------------------------------------------------------
#     # Gym API：step(action)
#     # ----------------------------------------------------------------------
#     def step(self, action):
#         """执行动作，返回 (next_state, reward, done, info)"""
#         x, y = divmod(self.agent_state, self.columns)
#         dx, dy = self.ACTIONS[action]
#         nx, ny = x + dx, y + dy

#         # 撞墙：固定惩罚，状态不变
#         if not (0 <= nx < self.rows and 0 <= ny < self.columns):
#             next_state = self.agent_state
#             reward = -1
#         else:
#             next_state = nx * self.columns + ny
#             reward = self.score_map[nx][ny]

#         self.agent_state = next_state

#         # 是否终止（如果走到 target）
#         done = next_state in self.terminal_states

#         return next_state, reward, done, {}
