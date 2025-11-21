import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2TagEnv:
    """鬼ごっこ環境: 追跡者（鬼）から逃げるロボット"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, chaser_policy=None):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        # 追跡者用のポリシー
        self.chaser_policy = chaser_policy

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(5.0, 0.0, 4.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=50,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=50,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add evader robot (逃げるロボット)
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # add chaser robot (追いかけるロボット - 鬼)
        self.chaser_init_pos = torch.tensor(self.env_cfg["chaser_init_pos"], device=gs.device)
        self.chaser_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.chaser = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.chaser_init_pos.cpu().numpy(),
                quat=self.chaser_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices for evader
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # names to indices for chaser
        self.chaser_motors_dof_idx = [self.chaser.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters for evader
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # PD control parameters for chaser
        self.chaser.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.chaser_motors_dof_idx)
        self.chaser.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.chaser_motors_dof_idx)

        # prepare reward functions
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers for evader
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )

        # initialize buffers for chaser
        self.chaser_base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.chaser_base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.chaser_dof_vel = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.chaser_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.chaser_commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.chaser_obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        # 追跡者との相対距離
        self.relative_pos_to_chaser = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.distance_to_chaser = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _update_chaser_behavior(self):
        """追跡者（鬼）の行動: 学習済みポリシーで逃走者を追いかける"""
        if self.chaser_policy is None:
            return

        # 相対位置を計算（逃走者の方向）
        relative_pos = self.base_pos - self.chaser_base_pos
        self.distance_to_chaser[:] = torch.norm(relative_pos[:, :2], dim=1)

        # 追跡者のコマンドを設定（逃走者の方向に向かう）
        direction = relative_pos[:, :2]
        direction_norm = torch.norm(direction, dim=1, keepdim=True)
        direction_normalized = direction / (direction_norm + 1e-6)

        # 追跡速度を設定
        chaser_speed = self.env_cfg.get("chaser_speed", 0.8)

        # コマンドベクトルを設定（x方向の速度、y方向の速度、角速度）
        self.chaser_commands[:, 0] = direction_normalized[:, 0] * chaser_speed
        self.chaser_commands[:, 1] = direction_normalized[:, 1] * chaser_speed
        self.chaser_commands[:, 2] = 0.0  # 角速度は0

        # 観測を構築（逃走者と同じ形式）
        self.chaser_obs_buf[:] = torch.cat(
            [
                self.chaser_base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.chaser_projected_gravity,  # 3
                self.chaser_commands * self.commands_scale,  # 3
                (self.chaser_dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.chaser_dof_vel * self.obs_scales["dof_vel"],  # 12
                self.chaser_actions,  # 12
            ],
            axis=-1,
        )

        # ポリシーからアクションを取得
        with torch.no_grad():
            self.chaser_actions[:] = self.chaser_policy(self.chaser_obs_buf)

        # 追跡者を制御
        exec_chaser_actions = self.chaser_actions
        target_chaser_dof_pos = exec_chaser_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        try:
            self.chaser.control_dofs_position(target_chaser_dof_pos, self.chaser_motors_dof_idx)
        except Exception as e:
            # インデックスエラーの場合、全自由度に対して制御を試みる
            print(f"Warning: Could not control chaser with specific indices: {e}")
            # 全自由度のターゲットを作成（最初の12個をターゲット、残りは現在値）
            all_target = self.chaser.get_dofs_position()
            all_target[:, :self.num_actions] = target_chaser_dof_pos
            self.chaser.control_dofs_position(all_target)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)

        # update buffers for chaser (before controlling)
        self.chaser_base_pos[:] = self.chaser.get_pos()
        self.chaser_base_quat[:] = self.chaser.get_quat()
        inv_chaser_quat = inv_quat(self.chaser_base_quat)
        self.chaser_base_lin_vel[:] = transform_by_quat(self.chaser.get_vel(), inv_chaser_quat)
        self.chaser_base_ang_vel[:] = transform_by_quat(self.chaser.get_ang(), inv_chaser_quat)
        self.chaser_projected_gravity[:] = transform_by_quat(self.global_gravity, inv_chaser_quat)

        # 追跡者の関節情報を取得（全自由度から該当部分を抽出）
        try:
            all_chaser_dofs = self.chaser.get_dofs_position()
            # モーターに対応する関節のみを抽出（最初の12個と仮定）
            self.chaser_dof_pos[:] = all_chaser_dofs[:, :self.num_actions]
            all_chaser_vel = self.chaser.get_dofs_velocity()
            self.chaser_dof_vel[:] = all_chaser_vel[:, :self.num_actions]
        except Exception as e:
            # エラーが発生した場合はデフォルト値を使用
            print(f"Warning: Could not get chaser dof state: {e}")
            self.chaser_dof_pos[:] = self.default_dof_pos
            self.chaser_dof_vel[:] = 0.0

        # 追跡者を制御
        self._update_chaser_behavior()

        self.scene.step()

        # update buffers for evader
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # update chaser position again after step
        self.chaser_base_pos[:] = self.chaser.get_pos()
        self.chaser_base_quat[:] = self.chaser.get_quat()

        # 相対位置を更新
        relative_pos = self.base_pos - self.chaser_base_pos
        self.distance_to_chaser[:] = torch.norm(relative_pos[:, :2], dim=1)
        self.relative_pos_to_chaser[:] = transform_by_quat(relative_pos, inv_base_quat)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        # 捕まった場合（追跡者が近すぎる場合）
        caught_distance = self.env_cfg.get("caught_distance", 0.5)
        self.reset_buf |= self.distance_to_chaser < caught_distance

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations (追跡者の相対位置を含む)
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.relative_pos_to_chaser * self.obs_scales["lin_vel"],  # 3 (追跡者の相対位置)
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset evader dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset evader base (ランダムな位置に配置)
        random_offset = torch.zeros((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float)
        random_offset[:, 0] = gs_rand_float(-2.0, 2.0, (len(envs_idx),), gs.device)
        random_offset[:, 1] = gs_rand_float(-2.0, 2.0, (len(envs_idx),), gs.device)

        self.base_pos[envs_idx] = self.base_init_pos + random_offset
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset chaser dofs
        self.chaser_dof_pos[envs_idx] = self.default_dof_pos
        self.chaser_dof_vel[envs_idx] = 0.0
        try:
            self.chaser.set_dofs_position(
                position=self.chaser_dof_pos[envs_idx],
                dofs_idx_local=self.chaser_motors_dof_idx,
                zero_velocity=True,
                envs_idx=envs_idx,
            )
        except Exception as e:
            # インデックスエラーの場合、全自由度をリセット
            print(f"Warning: Could not reset chaser dofs with specific indices: {e}")
            all_dofs = self.chaser.get_dofs_position()
            all_dofs[envs_idx, :self.num_actions] = self.default_dof_pos
            self.chaser.set_dofs_position(position=all_dofs[envs_idx], zero_velocity=True, envs_idx=envs_idx)

        # reset chaser base (逃走者から一定距離離れた位置)
        chaser_offset = torch.zeros((len(envs_idx), 3), device=gs.device, dtype=gs.tc_float)
        chaser_offset[:, 0] = gs_rand_float(-3.0, -2.0, (len(envs_idx),), gs.device)
        chaser_offset[:, 1] = gs_rand_float(-1.0, 1.0, (len(envs_idx),), gs.device)

        self.chaser_base_pos[envs_idx] = self.chaser_init_pos + chaser_offset
        self.chaser_base_quat[envs_idx] = self.chaser_init_quat.reshape(1, -1)
        self.chaser.set_pos(self.chaser_base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.chaser.set_quat(self.chaser_base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.chaser_base_lin_vel[envs_idx] = 0
        self.chaser_base_ang_vel[envs_idx] = 0
        self.chaser.zero_all_dofs_velocity(envs_idx)
        self.chaser_actions[envs_idx] = 0.0
        self.chaser_commands[envs_idx] = 0.0

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_avoid_chaser(self):
        """追跡者から離れることに報酬"""
        return torch.clamp(self.distance_to_chaser, 0, 5.0) / 5.0

    def _reward_forward_vel(self):
        """前進速度に報酬"""
        return torch.clamp(self.base_lin_vel[:, 0], 0, 2.0)

    def _reward_survival(self):
        """生存に報酬"""
        return torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_lin_vel_z(self):
        """z軸方向の速度にペナルティ"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        """アクションの変化にペナルティ"""
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """デフォルト姿勢から離れることにペナルティ"""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        """目標高さから離れることにペナルティ"""
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
