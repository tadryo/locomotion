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

from go2_tag_env import Go2TagEnv


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
        # termination
        "termination_if_roll_greater_than": 30,  # degree (より寛容に)
        "termination_if_pitch_greater_than": 30,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # chaser settings
        "chaser_init_pos": [-3.0, 0.0, 0.42],
        "chaser_speed": 0.6,  # 追跡者の速度（0.6 m/s）
        "caught_distance": 0.5,  # 捕まる距離（0.5m以内）
        "episode_length_s": 20.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,  # 3 (ang_vel) + 3 (gravity) + 3 (relative_pos) + 12 (dof_pos) + 12 (dof_vel) + 12 (actions)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    # 鬼ごっこ用の報酬設定
    reward_cfg = {
        "base_height_target": 0.35,
        "reward_scales": {
            "avoid_chaser": 5.0,  # 追跡者から離れることへの報酬
            "forward_vel": 1.0,  # 前進速度への報酬
            "survival": 0.5,  # 生存報酬
            "lin_vel_z": -1.0,  # z軸速度へのペナルティ
            "base_height": -10.0,  # 高さ維持
            "action_rate": -0.01,  # アクション変化のペナルティ
            "similar_to_default": -0.05,  # デフォルト姿勢からのペナルティ
        },
    }
    command_cfg = {
        "num_commands": 3,  # 使わないが互換性のために保持
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-tag-game")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--load_exp_name", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--show_viewer", action="store_true", help="Show viewer for debugging")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2TagEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    if args.load_exp_name:
        model_path = f"logs/{args.load_exp_name}/model_{args.ckpt}.pt"
        print(f"Loading pretrained model: {model_path}")
        if os.path.exists(model_path):
            runner.load(model_path)
        else:
            print(f"Error: Model not found: {model_path}")
            exit(1)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    # macOS notification with sound
    pf = system()
    if pf == "Darwin":
        os.system("afplay /System/Library/Sounds/Blow.aiff")
        os.system(
            f"osascript -e 'display notification \"Experiment: {args.exp_name}\" with title \"Go2 Tag Game Training Completed\"'"
        )


if __name__ == "__main__":
    main()


"""
# training
python go2_tag_train.py --exp_name go2-tag-game --num_envs 4096 --max_iterations 500

# training with viewer (for debugging)
python go2_tag_train.py --exp_name go2-tag-game --num_envs 16 --max_iterations 10 --show_viewer

# load pretrained model and continue training
python go2_tag_train.py --exp_name go2-tag-game-v2 --load_exp_name go2-tag-game --ckpt 500
"""
