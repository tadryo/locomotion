import argparse
import os
import pickle
import shutil
from importlib import metadata

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


def get_train_cfg(exp_name, max_iterations):
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
        "save_interval": 100,
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
        # termination - 走行時はより大きな傾きを許容
        "termination_if_roll_greater_than": 15,  # degree
        "termination_if_pitch_greater_than": 15,
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
    # 走行用報酬設定
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            # 基本的な報酬
            "tracking_lin_vel": 1.5,  # 速度追跡を強化
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -0.5,  # 垂直方向の動きのペナルティを軽減
            "base_height": -30.0,  # 高さのペナルティを軽減
            "action_rate": -0.002,  # アクション変化のペナルティを軽減
            "similar_to_default": -0.05,  # デフォルト姿勢からの乖離ペナルティを軽減
            # カスタム報酬（走行特化）
            "forward_distance": 10.0,  # 前進距離を強く報酬化
            "diagonal_gait": 0.5,  # 対角歩容の奨励
            "aligned_hips": 0.3,  # ヒップ関節の整列
            "straight_line": 0.5,  # 直進性の維持
            "foot_clearance": 0.2,  # 足の持ち上げ
            "energy_efficiency": -0.001,  # エネルギー効率（ペナルティ）
        },
    }
    # 走行用コマンド設定：より高速な前進
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [1.5, 2.0],  # 高速前進
        "lin_vel_y_range": [0.0, 0.0],  # 横方向の動きなし
        "ang_vel_range": [0.0, 0.0],  # 回転なし
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-running")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="続きから学習を再開")
    parser.add_argument("--ckpt", type=int, default=100, help="再開するチェックポイント番号")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"

    if args.resume:
        # 続きから学習：既存の設定とモデルをロード
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
        train_cfg["runner"]["max_iterations"] = args.max_iterations
    else:
        # 新規学習：設定を生成してログディレクトリを初期化
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

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

    if args.resume:
        # チェックポイントをロードして続きから学習
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        print(f"チェックポイント {args.ckpt} から学習を再開します: {resume_path}")
        runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# 走行動作のトレーニング（新規）
python go2_running_train.py -e go2-running --max_iterations 500

# カスタムパラメータでのトレーニング
python go2_running_train.py -e go2-running -B 2048 --max_iterations 1000

# 続きから学習を再開（model_100.ptから500イテレーションまで）
python go2_running_train.py -e go2-running --resume --ckpt 100 --max_iterations 500

# 別のチェックポイントから再開
python go2_running_train.py -e go2-running --resume --ckpt 200 --max_iterations 1000
"""
