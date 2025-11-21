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

from go2_tag_env import Go2TagEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-tag-game")
    parser.add_argument("--ckpt", type=int, default=500)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # Load chaser policy
    chaser_policy = None
    chaser_model_path = "logs/go2-walking-0.5/model_100.pt"
    if os.path.exists(chaser_model_path):
        from rsl_rl.modules import ActorCritic

        policy_cfg = train_cfg["policy"].copy()
        policy_cfg.pop("class_name", None)  # class_nameを除外
        temp_policy = ActorCritic(obs_cfg["num_obs"], None, env_cfg["num_actions"], **policy_cfg).to(gs.device)
        loaded_dict = torch.load(chaser_model_path)
        temp_policy.load_state_dict(loaded_dict["model_state_dict"])
        temp_policy.eval()
        chaser_policy = temp_policy.act_inference
        print(f"Chaser policy loaded from: {chaser_model_path}")
    else:
        print(f"Warning: Chaser model not found: {chaser_model_path}")

    # Create environment with viewer
    env = Go2TagEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        chaser_policy=chaser_policy,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    print(f"Loading model from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # Evaluation loop
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rew, done, info = env.step(actions)

            # Print distance info
            if env.episode_length_buf[0] % 50 == 0:
                distance = env.distance_to_chaser[0].item()
                print(f"Step: {env.episode_length_buf[0].item()}, Distance to chaser: {distance:.2f}m")


if __name__ == "__main__":
    main()


"""
# evaluation
python go2_tag_eval.py --exp_name go2-tag-game --ckpt 500
"""
