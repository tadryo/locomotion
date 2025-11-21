import argparse
import os
import pickle

import genesis as gs
import torch

from go2_tag_env import Go2TagEnv


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-tag-game")
    parser.add_argument("--ckpt", type=int, default=500)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))

    # Create environment with viewer
    env = Go2TagEnv(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
    )

    # Load policy
    from rsl_rl.modules import ActorCritic

    policy_cfg = train_cfg["policy"]
    policy = ActorCritic(env.num_obs, env.num_privileged_obs, env.num_actions, **policy_cfg).to(env.device)
    model_path = f"{log_dir}/model_{args.ckpt}.pt"
    print(f"Loading model from: {model_path}")
    loaded_dict = torch.load(model_path)
    policy.load_state_dict(loaded_dict["model_state_dict"])
    policy.eval()

    # Evaluation loop
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy.act_inference(obs)
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
