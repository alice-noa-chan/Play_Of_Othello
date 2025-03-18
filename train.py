import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from tqdm.auto import tqdm
import os

# --- 한 단계 탐색을 위한 시뮬레이션 함수 ---
def simulate_move(board, move, player, board_size=8):
    """
    주어진 board 복사본에 대해 move를 적용하고, 돌 뒤집기를 수행.
    board는 이미 복사본이어야 합니다.
    """
    i, j = move
    board[i, j] = player
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    for d in directions:
        discs_to_flip = []
        ni, nj = i + d[0], j + d[1]
        while 0 <= ni < board_size and 0 <= nj < board_size:
            if board[ni, nj] == -player:
                discs_to_flip.append((ni, nj))
            elif board[ni, nj] == player:
                for (fi, fj) in discs_to_flip:
                    board[fi, fj] = player
                break
            else:
                break
            ni += d[0]
            nj += d[1]
    return board

# --- 커스텀 리버시 환경 (self-play, 탐색 기법 추가 및 reward shaping) ---
class ReversiEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, opponent_policy=None):
        """
        opponent_policy: 상대의 행동 결정을 위한 PPO 모델 (또는 유사 객체).
        없으면 random opponent를 사용합니다.
        """
        super(ReversiEnv, self).__init__()
        self.board_size = 8
        # 0~63: 각 셀에 돌을 놓는 액션, 64: 패스 액션
        self.action_space = spaces.Discrete(self.board_size * self.board_size + 1)
        # 보드는 -1 (상대), 0 (빈 칸), 1 (내 돌)로 표시
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.opponent_policy = opponent_policy
        
        # 보상 shaping 관련 파라미터
        self.incremental_scale = 0.5    # 돌 개수 변화 보상 계수
        self.illegal_penalty = -2       # 불법 행동 패널티
        self.final_reward_scale = 1.5   # 최종 승패 보상 가중치
        
        self.reset()

    def reset(self, **kwargs):
        # 초기 보드 상태: 중앙 4칸 배치
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        mid = self.board_size // 2
        self.board[mid-1, mid-1] = -1
        self.board[mid, mid] = -1
        self.board[mid-1, mid] = 1
        self.board[mid, mid-1] = 1
        self.current_player = 1  # 에이전트: 1, 상대: -1
        self.done = False
        return self.board.copy(), {}

    def step(self, action):
        # 현재 스텝 시작 전, 에이전트의 돌 개수 차이 (보상 shaping용)
        pre_diff = np.sum(self.board == 1) - np.sum(self.board == -1)
        info = {}
        
        if self.done:
            return self.board.copy(), 0, True, False, info

        # 에이전트 차례 (항상 player 1)
        if self.current_player != 1:
            raise Exception("현재 에이전트의 차례가 아닙니다.")
        
        valid_moves = self.get_valid_moves(self.current_player)
        
        # 패스 액션 처리
        if action == self.board_size * self.board_size:
            if valid_moves:
                # 합법 수가 있음에도 패스 선택 시, 무작위 합법 수 적용 + 패널티
                move = random.choice(valid_moves)
                penalty = self.illegal_penalty
                info["illegal_move"] = True
                info["correction"] = "pass->move"
            else:
                # 합법 수가 없으면 패스 허용
                move = None
                penalty = 0
        else:
            move = (action // self.board_size, action % self.board_size)
            if move not in valid_moves:
                # 불법 수 선택 시, 무작위 합법 수 적용 + 패널티
                if valid_moves:
                    move = random.choice(valid_moves)
                    penalty = self.illegal_penalty
                    info["illegal_move"] = True
                    info["correction"] = "illegal->move"
                else:
                    # 합법 수가 없으면 패스
                    move = None
                    penalty = 0
            else:
                penalty = 0
        
        # 에이전트의 수행: 수 선택 (패스가 아니라면)
        if move is not None:
            self.make_move(move, self.current_player)
        
        # 게임 종료 여부 확인 (에이전트 수 이후)
        if self.is_game_over():
            final_reward = self.final_reward_scale * self.get_reward()  # 최종 보상에 가중치 적용
            self.done = True
            return self.board.copy(), final_reward, True, False, info
        
        # --- 상대(self-play) 차례 ---
        opponent_valid_moves = self.get_valid_moves(-self.current_player)
        if opponent_valid_moves:
            if self.opponent_policy is not None:
                opp_move = self.get_opponent_move(opponent_valid_moves)
            else:
                opp_move = random.choice(opponent_valid_moves)
            self.make_move(opp_move, -self.current_player)
        
        # 상대 수 이후 게임 종료 여부 확인
        if self.is_game_over():
            final_reward = self.final_reward_scale * self.get_reward()
            self.done = True
            return self.board.copy(), final_reward, True, False, info
        
        # 보상 shaping: 에이전트 돌 개수 차이 변화 (변화량에 incremental_scale 적용)
        post_diff = np.sum(self.board == 1) - np.sum(self.board == -1)
        incremental_reward = self.incremental_scale * (post_diff - pre_diff)
        
        # 불법 수에 대한 패널티 반영
        total_reward = penalty + incremental_reward
        
        # 다음 에이전트 차례로 전환
        self.current_player = 1
        return self.board.copy(), total_reward, False, False, info

    def get_valid_moves(self, player):
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0 and self.is_valid_move((i, j), player):
                    valid_moves.append((i, j))
        return valid_moves

    def is_valid_move(self, move, player):
        i, j = move
        if self.board[i, j] != 0:
            return False
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        for d in directions:
            ni, nj = i + d[0], j + d[1]
            has_opponent_between = False
            while 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                if self.board[ni, nj] == -player:
                    has_opponent_between = True
                elif self.board[ni, nj] == player:
                    if has_opponent_between:
                        return True
                    break
                else:
                    break
                ni += d[0]
                nj += d[1]
        return False

    def make_move(self, move, player):
        i, j = move
        self.board[i, j] = player
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        for d in directions:
            discs_to_flip = []
            ni, nj = i + d[0], j + d[1]
            while 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                if self.board[ni, nj] == -player:
                    discs_to_flip.append((ni, nj))
                elif self.board[ni, nj] == player:
                    for (fi, fj) in discs_to_flip:
                        self.board[fi, fj] = player
                    break
                else:
                    break
                ni += d[0]
                nj += d[1]

    def get_opponent_move(self, valid_moves):
        """
        각 유효 수에 대해 시뮬레이션한 후, opponent_policy의 predict_values 메서드를 이용해 평가하여
        가장 좋은 수를 선택합니다.
        """
        best_move = None
        best_value = -float('inf')
        for move in valid_moves:
            board_copy = self.board.copy()
            board_copy = simulate_move(board_copy, move, -self.current_player, self.board_size)
            obs = board_copy.flatten().astype(np.float32)
            obs_tensor = torch.tensor(obs).unsqueeze(0).to(next(self.opponent_policy.policy.parameters()).device)
            with torch.no_grad():
                value = self.opponent_policy.policy.predict_values(obs_tensor).item()
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

    def is_game_over(self):
        # 양쪽 모두 둘 곳이 없으면 종료
        if len(self.get_valid_moves(1)) == 0 and len(self.get_valid_moves(-1)) == 0:
            return True
        return False

    def get_reward(self):
        # 최종 보상: 에이전트 관점에서 승리 +1, 패배 -1, 무승부 0
        agent_count = np.sum(self.board == 1)
        opp_count = np.sum(self.board == -1)
        if agent_count > opp_count:
            return 1
        elif agent_count < opp_count:
            return -1
        else:
            return 0

    def render(self, mode="human"):
        symbols = {1: "●", -1: "○", 0: "."}
        board_str = "\n".join(" ".join(symbols[cell] for cell in row) for row in self.board)
        print(board_str)
        print()


# --- tqdm progress bar를 위한 Callback ---
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TqdmCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.tqdm_bar = tqdm(total=self.total_timesteps)

    def _on_step(self):
        self.tqdm_bar.update(1)
        return True

    def _on_training_end(self):
        self.tqdm_bar.close()


# --- 일정 주기마다 모델 저장 Callback ---
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_filename = os.path.join(self.save_path, f"model_step_{self.n_calls}.zip")
            self.model.save(model_filename)
            if self.verbose:
                print(f"Saved model at step {self.n_calls} to {model_filename}")
        return True


# --- 주기적으로 opponent_policy 업데이트 Callback ---
class OpponentUpdateCallback(BaseCallback):
    def __init__(self, update_freq, env, verbose=0):
        super(OpponentUpdateCallback, self).__init__(verbose)
        self.update_freq = update_freq
        self.env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.update_freq == 0:
            self.env.opponent_policy = self.model
            if self.verbose:
                print(f"Updated opponent policy at step {self.n_calls}")
        return True


# --- 학습 실행 ---
if __name__ == "__main__":
    # 총 학습 타임스텝을 400,000으로 설정
    total_timesteps = 400000

    # CPU 사용 (GPU 대신)
    device = "cpu"
    
    env = ReversiEnv(opponent_policy=None)

    # 재현성을 위한 시드 설정
    seed = 42
    env.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    MODEL_PATH = "reversi_ai_model_final.zip"
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, env=env, device=device)
        env.opponent_policy = model
        print("모델 파일이 존재하여 추가 학습을 진행합니다.")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs
        )

    tqdm_callback = TqdmCallback(total_timesteps, verbose=1)
    save_model_callback = SaveModelCallback(save_freq=5000, save_path="models", verbose=1)
    opponent_update_callback = OpponentUpdateCallback(update_freq=5000, env=env, verbose=1)
    callback = CallbackList([tqdm_callback, save_model_callback, opponent_update_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("reversi_ai_model_final")
    print("학습 완료 및 최종 모델 저장!")
