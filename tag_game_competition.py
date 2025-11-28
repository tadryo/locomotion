import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from tag_game_env import TagGameEnv


class CompetitionEnv(TagGameEnv):
    """コンペティション用環境 - 両方のロボットが学習済みモデルを使用"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.match_ended = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self.final_survival_time = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def step(self, runner_actions, chaser_actions):
        """両方のロボットのアクションを受け取り、シミュレーションを進める"""
        # runner actions
        self.runner_actions = torch.clip(runner_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_runner_actions = self.runner_last_actions if self.simulate_action_latency else self.runner_actions
        target_runner_dof_pos = exec_runner_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.runner.control_dofs_position(target_runner_dof_pos, self.motors_dof_idx)

        # chaser actions
        self.chaser_actions = torch.clip(chaser_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_chaser_actions = self.chaser_last_actions if self.simulate_action_latency else self.chaser_actions
        target_chaser_dof_pos = exec_chaser_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.chaser.control_dofs_position(target_chaser_dof_pos, self.motors_dof_idx)

        # simulation step
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        # 終了していない環境のみ生存時間を加算
        self.survival_time += self.dt * (~self.match_ended).float()

        self._update_robot_state(self.runner, "runner")
        self._update_robot_state(self.chaser, "chaser")

        # 捕獲チェック
        distance = self._compute_distance()
        newly_caught = (distance < self.catch_distance) & (~self.match_ended)

        # 捕獲された時の生存時間を記録
        self.final_survival_time[newly_caught] = self.survival_time[newly_caught]
        self.match_ended |= newly_caught
        self.caught = distance < self.catch_distance

        # check termination
        self.reset_buf = self.episode_length_buf > self.max_episode_length

        # タイムアウトした環境の生存時間を記録
        timed_out = (self.episode_length_buf > self.max_episode_length) & (~self.match_ended)
        self.final_survival_time[timed_out] = self.survival_time[timed_out]
        self.match_ended |= timed_out

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        # compute observations
        runner_obs = self._compute_obs("runner", self.runner_actions)
        chaser_obs = self._compute_obs("chaser", self.chaser_actions)

        self.runner_last_actions[:] = self.runner_actions[:]
        self.chaser_last_actions[:] = self.chaser_actions[:]

        return runner_obs, chaser_obs, self.reset_buf, self.match_ended, self.final_survival_time

    def reset_idx(self, envs_idx):
        super().reset_idx(envs_idx)
        self.match_ended[envs_idx] = False
        self.final_survival_time[envs_idx] = 0.0


def load_policy(exp_name, ckpt, env, device):
    """学習済みモデルをロード"""
    log_dir = f"logs/{exp_name}"
    model_path = os.path.join(log_dir, f"model_{ckpt}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 設定を読み込み
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    if os.path.exists(cfg_path):
        _, _, _, _, train_cfg = pickle.load(open(cfg_path, "rb"))
    else:
        # デフォルトの設定を使用
        train_cfg = {
            "algorithm": {
                "class_name": "PPO",
                "clip_param": 0.2,
                "desired_kl": 0.01,
                "entropy_coef": 0.01,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 0.001,
                "max_grad_norm": 1.0,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "schedule": "adaptive",
                "use_clipped_value_loss": True,
                "value_loss_coef": 1.0,
            },
            "init_member_classes": {},
            "policy": {
                "activation": "elu",
                "actor_hidden_dims": [512, 256, 128],
                "critic_hidden_dims": [512, 256, 128],
                "init_noise_std": 1.0,
                "class_name": "ActorCritic",
            },
            "runner": {
                "checkpoint": -1,
                "experiment_name": exp_name,
                "load_run": -1,
                "log_interval": 1,
                "max_iterations": 500,
                "record_interval": -1,
                "resume": False,
                "resume_path": None,
                "run_name": "",
            },
            "runner_class_name": "OnPolicyRunner",
            "num_steps_per_env": 24,
            "save_interval": 100,
            "empirical_normalization": None,
            "seed": 1,
        }

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=device)

    return policy


def get_competition_cfgs():
    """コンペティション用の環境設定"""
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 30,
        "termination_if_pitch_greater_than": 30,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "runner_init_pos": [-2.0, 0.0, 0.42],
        "chaser_init_pos": [2.0, 0.0, 0.42],
        "episode_length_s": 30.0,  # コンペティションは30秒
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "field_size": 10.0,
        "wall_height": 1.0,
        "wall_thickness": 0.1,
        "catch_distance": 0.5,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {},  # コンペティションでは報酬なし
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def run_match(env, runner_policy, chaser_policy, num_trials):
    """対戦を実行し、生存時間の平均を返す"""
    survival_times = []

    for trial in range(num_trials):
        env.reset()

        # runner と chaser の観測を初期化
        env._update_robot_state(env.runner, "runner")
        env._update_robot_state(env.chaser, "chaser")
        runner_obs = env._compute_obs("runner", env.runner_actions)
        chaser_obs = env._compute_obs("chaser", env.chaser_actions)

        with torch.no_grad():
            while not env.match_ended.all():
                runner_actions = runner_policy(runner_obs)
                chaser_actions = chaser_policy(chaser_obs)

                runner_obs, chaser_obs, reset_buf, match_ended, final_times = env.step(
                    runner_actions, chaser_actions
                )

                if match_ended.all():
                    break

        # 生存時間を記録
        avg_survival = env.final_survival_time.mean().item()
        survival_times.append(avg_survival)
        print(f"  Trial {trial + 1}: Average survival time = {avg_survival:.2f}s")

        # リセット
        env.reset()

    return sum(survival_times) / len(survival_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_a_runner", type=str, required=True, help="Team A runner exp_name")
    parser.add_argument("--team_a_runner_ckpt", type=int, default=500)
    parser.add_argument("--team_a_chaser", type=str, required=True, help="Team A chaser exp_name")
    parser.add_argument("--team_a_chaser_ckpt", type=int, default=500)
    parser.add_argument("--team_b_runner", type=str, required=True, help="Team B runner exp_name")
    parser.add_argument("--team_b_runner_ckpt", type=int, default=500)
    parser.add_argument("--team_b_chaser", type=str, required=True, help="Team B chaser exp_name")
    parser.add_argument("--team_b_chaser_ckpt", type=int, default=500)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("-B", "--num_envs", type=int, default=100)
    parser.add_argument("--show_viewer", action="store_true")
    args = parser.parse_args()

    gs.init()

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_competition_cfgs()

    # 環境を作成
    env = CompetitionEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    print("=" * 60)
    print("Tag Game Competition")
    print("=" * 60)

    # ポリシーをロード
    print("\nLoading policies...")
    team_a_runner_policy = load_policy(args.team_a_runner, args.team_a_runner_ckpt, env, gs.device)
    team_a_chaser_policy = load_policy(args.team_a_chaser, args.team_a_chaser_ckpt, env, gs.device)
    team_b_runner_policy = load_policy(args.team_b_runner, args.team_b_runner_ckpt, env, gs.device)
    team_b_chaser_policy = load_policy(args.team_b_chaser, args.team_b_chaser_ckpt, env, gs.device)

    # 対戦1: Team A の chaser vs Team B の runner
    print("\n" + "-" * 60)
    print("Match 1: Team A (Chaser) vs Team B (Runner)")
    print("-" * 60)
    team_b_survival = run_match(env, team_b_runner_policy, team_a_chaser_policy, args.num_trials)
    print(f"Team B average survival time: {team_b_survival:.2f}s")

    # 対戦2: Team B の chaser vs Team A の runner
    print("\n" + "-" * 60)
    print("Match 2: Team B (Chaser) vs Team A (Runner)")
    print("-" * 60)
    team_a_survival = run_match(env, team_a_runner_policy, team_b_chaser_policy, args.num_trials)
    print(f"Team A average survival time: {team_a_survival:.2f}s")

    # 結果
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Team A total score: {team_a_survival:.2f}s")
    print(f"Team B total score: {team_b_survival:.2f}s")
    print()

    if team_a_survival > team_b_survival:
        print("Winner: Team A!")
    elif team_b_survival > team_a_survival:
        print("Winner: Team B!")
    else:
        print("Draw!")

    print("=" * 60)


if __name__ == "__main__":
    main()


"""
# コンペティション
python tag_game_competition.py \
    --team_a_runner team_a_runner --team_a_runner_ckpt 500 \
    --team_a_chaser team_a_chaser --team_a_chaser_ckpt 500 \
    --team_b_runner team_b_runner --team_b_runner_ckpt 500 \
    --team_b_chaser team_b_chaser --team_b_chaser_ckpt 500 \
    --num_trials 10
"""
