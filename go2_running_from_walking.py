import argparse
import os
import pickle
import shutil
from importlib import metadata
from platform import system

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

from go2_running_env import Go2RunningEnv


def get_train_cfg(exp_name, max_iterations, save_interval=100):
    train_cfg_dict = {
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
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": save_interval,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
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
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
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
    # 走行用報酬設定（歩行ベース + ギャロップ歩容 + 直進性強化）
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            # 基本的な報酬
            "tracking_lin_vel": 1.5,  # 速度追跡を強化
            "tracking_ang_vel": 1.5,  # 角速度追跡を強化（回転を抑制）
            "lin_vel_z": -2.0,  # Z軸ペナルティ強化（跳ねすぎ防止）
            "base_height": -50.0,  # 歩行と同じ
            "action_rate": -0.01,  # アクション変化ペナルティ強化（滑らかな動き）
            "similar_to_default": -0.2,  # デフォルト姿勢を重視（脚を上げすぎ防止）
            # カスタム報酬（ギャロップ + 直進性）
            "gallop_gait": 1.0,  # ギャロップ歩容の奨励
            "straight_line": 1.5,  # 直進性を強く奨励（クルクル回る問題を解決）
            # 以下は無効化
            # "diagonal_gait": 0.5,  # トロット用（ギャロップには不適）
            # "forward_distance": 4.0,
            # "aligned_hips": 0.3,
            # "foot_clearance": 0.3,  # 脚を上げすぎる原因
            # "energy_efficiency": -0.002,
        },
    }
    # 走行用コマンド設定：段階的に速度を上げる
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [10.0, 10.0],  # まずは2.5 m/sを目指す
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],  # 回転なし
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-running-from-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--walking_model", type=str, default="logs/go2-walking/model_100.pt",
                        help="歩行モデルのパス")
    parser.add_argument("--save_every_step", action="store_true", help="1イテレーションごとにモデルを保存")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"

    # save_intervalを設定
    save_interval = 1 if args.save_every_step else 100

    # 新規学習：設定を生成してログディレクトリを初期化
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, save_interval)

    if args.save_every_step:
        print("1イテレーションごとにモデルを保存します")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2RunningEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # 歩行モデルをロード
    if os.path.exists(args.walking_model):
        print(f"歩行モデルをロード: {args.walking_model}")
        runner.load(args.walking_model)
        print("歩行モデルから走行学習を開始します")
    else:
        print(f"警告: 歩行モデルが見つかりません: {args.walking_model}")
        print("ゼロから学習を開始します")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # トレーニング終了通知
    print(f"\n{'='*60}")
    print(f"トレーニングが完了しました！")
    print(f"実験名: {args.exp_name}")
    print(f"最大イテレーション: {args.max_iterations}")
    print(f"ログディレクトリ: {log_dir}")
    print(f"{'='*60}\n")

    # macOSの場合は通知を表示
    pf = system()
    if pf == "Darwin":
        os.system(f"osascript -e 'display notification \"実験名: {args.exp_name}\" with title \"Go2 トレーニング完了\"'")


if __name__ == "__main__":
    main()

"""
# 歩行モデルから走行学習（推奨）
python go2_running_from_walking.py -e go2-running-from-walking --max_iterations 300

# カスタム歩行モデルから開始
python go2_running_from_walking.py -e go2-running-from-walking --walking_model logs/go2-walking/model_100.pt --max_iterations 500

# 1イテレーションごとに保存
python go2_running_from_walking.py -e go2-running-from-walking --max_iterations 100 --save_every_step
"""
