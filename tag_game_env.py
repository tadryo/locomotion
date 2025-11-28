import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class TagGameEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, runner_policy=None, chaser_policy=None):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # field parameters
        self.field_size = env_cfg.get("field_size", 10.0)
        self.wall_height = env_cfg.get("wall_height", 1.0)
        self.wall_thickness = env_cfg.get("wall_thickness", 0.1)
        self.catch_distance = env_cfg.get("catch_distance", 0.5)

        # policies (not the ones being trained)
        self.runner_policy = runner_policy
        self.chaser_policy = chaser_policy

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(0.0, 0.0, 15.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=100,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add walls (4 sides)
        half_size = self.field_size / 2
        wall_positions = [
            [half_size + self.wall_thickness / 2, 0, self.wall_height / 2],  # +X wall
            [-half_size - self.wall_thickness / 2, 0, self.wall_height / 2],  # -X wall
            [0, half_size + self.wall_thickness / 2, self.wall_height / 2],  # +Y wall
            [0, -half_size - self.wall_thickness / 2, self.wall_height / 2],  # -Y wall
        ]
        wall_sizes = [
            [self.wall_thickness, self.field_size + 2 * self.wall_thickness, self.wall_height],  # +X
            [self.wall_thickness, self.field_size + 2 * self.wall_thickness, self.wall_height],  # -X
            [self.field_size + 2 * self.wall_thickness, self.wall_thickness, self.wall_height],  # +Y
            [self.field_size + 2 * self.wall_thickness, self.wall_thickness, self.wall_height],  # -Y
        ]

        self.walls = []
        for pos, size in zip(wall_positions, wall_sizes):
            wall = self.scene.add_entity(
                gs.morphs.Box(
                    pos=pos,
                    size=size,
                    fixed=True,
                    visualization=True,
                )
            )
            self.walls.append(wall)

        # add runner robot
        self.runner_init_pos = torch.tensor(env_cfg.get("runner_init_pos", [-2.0, 0.0, 0.42]), device=gs.device)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.runner = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.runner_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # add chaser robot
        self.chaser_init_pos = torch.tensor(env_cfg.get("chaser_init_pos", [2.0, 0.0, 0.42]), device=gs.device)
        self.chaser = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.chaser_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.runner.get_joint(name).dof_start for name in env_cfg["joint_names"]]

        # PD control parameters for both robots
        for robot in [self.runner, self.chaser]:
            robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
            robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers for runner
        self.runner_base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.runner_base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.runner_projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.runner_base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.runner_base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.runner_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.runner_dof_vel = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.runner_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.runner_last_actions = torch.zeros_like(self.runner_actions)

        # initialize buffers for chaser
        self.chaser_base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.chaser_base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.chaser_dof_pos = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.chaser_dof_vel = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.chaser_actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.chaser_last_actions = torch.zeros_like(self.chaser_actions)

        # shared buffers
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
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

        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        # wall collision buffers
        self.runner_wall_collision = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)
        self.chaser_wall_collision = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)

        # previous step distance (for approach reward)
        self.prev_distance = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # survival time
        self.survival_time = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # capture flag
        self.caught = torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)

        self.extras = dict()
        self.extras["observations"] = dict()

    def _compute_distance(self):
        """Compute distance between runner and chaser"""
        diff = self.runner_base_pos[:, :2] - self.chaser_base_pos[:, :2]
        return torch.norm(diff, dim=-1)

    def _check_wall_collision(self, pos):
        """Check collision with walls"""
        half_size = self.field_size / 2 - 0.3  # margin
        collision = (
            (pos[:, 0] > half_size)
            | (pos[:, 0] < -half_size)
            | (pos[:, 1] > half_size)
            | (pos[:, 1] < -half_size)
        )
        return collision

    def _update_robot_state(self, robot, prefix):
        """Update robot state buffers"""
        base_pos = robot.get_pos()
        base_quat = robot.get_quat()
        inv_base_quat = inv_quat(base_quat)
        base_lin_vel = transform_by_quat(robot.get_vel(), inv_base_quat)
        base_ang_vel = transform_by_quat(robot.get_ang(), inv_base_quat)
        projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        dof_pos = robot.get_dofs_position(self.motors_dof_idx)
        dof_vel = robot.get_dofs_velocity(self.motors_dof_idx)

        setattr(self, f"{prefix}_base_pos", base_pos)
        setattr(self, f"{prefix}_base_quat", base_quat)
        setattr(self, f"{prefix}_base_lin_vel", base_lin_vel)
        setattr(self, f"{prefix}_base_ang_vel", base_ang_vel)
        setattr(self, f"{prefix}_projected_gravity", projected_gravity)
        setattr(self, f"{prefix}_dof_pos", dof_pos)
        setattr(self, f"{prefix}_dof_vel", dof_vel)

        return base_pos, base_quat, base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel

    def _compute_obs(self, prefix, actions):
        """Compute observations (45 dimensions)"""
        base_ang_vel = getattr(self, f"{prefix}_base_ang_vel")
        projected_gravity = getattr(self, f"{prefix}_projected_gravity")
        dof_pos = getattr(self, f"{prefix}_dof_pos")
        dof_vel = getattr(self, f"{prefix}_dof_vel")

        obs = torch.cat(
            [
                base_ang_vel * self.obs_scales["ang_vel"],  # 3
                projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                dof_vel * self.obs_scales["dof_vel"],  # 12
                actions,  # 12
            ],
            axis=-1,
        )
        return obs

    def step(self, actions):
        """Advance the simulation by applying the actions of the robot being trained"""
        raise NotImplementedError("Use TagGameRunnerEnv or TagGameChaserEnv")

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset runner
        self.runner_dof_pos[envs_idx] = self.default_dof_pos
        self.runner_dof_vel[envs_idx] = 0.0
        self.runner.set_dofs_position(
            position=self.runner_dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # random initial position (runner)
        runner_x = gs_rand_float(-self.field_size / 3, self.field_size / 3, (len(envs_idx),), gs.device)
        runner_y = gs_rand_float(-self.field_size / 3, self.field_size / 3, (len(envs_idx),), gs.device)
        self.runner_base_pos[envs_idx, 0] = runner_x
        self.runner_base_pos[envs_idx, 1] = runner_y
        self.runner_base_pos[envs_idx, 2] = self.runner_init_pos[2]
        self.runner_base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.runner.set_pos(self.runner_base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.runner.set_quat(self.runner_base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.runner_base_lin_vel[envs_idx] = 0
        self.runner_base_ang_vel[envs_idx] = 0
        self.runner.zero_all_dofs_velocity(envs_idx)

        # reset chaser
        self.chaser_dof_pos[envs_idx] = self.default_dof_pos
        self.chaser_dof_vel[envs_idx] = 0.0
        self.chaser.set_dofs_position(
            position=self.chaser_dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # random initial position (chaser) - away from runner
        chaser_x = gs_rand_float(-self.field_size / 3, self.field_size / 3, (len(envs_idx),), gs.device)
        chaser_y = gs_rand_float(-self.field_size / 3, self.field_size / 3, (len(envs_idx),), gs.device)
        # keep at least 2m away from runner
        too_close = (torch.abs(chaser_x - runner_x) < 2.0) & (torch.abs(chaser_y - runner_y) < 2.0)
        chaser_x[too_close] = runner_x[too_close] + 2.5 * torch.sign(chaser_x[too_close] - runner_x[too_close] + 0.1)

        self.chaser_base_pos[envs_idx, 0] = chaser_x
        self.chaser_base_pos[envs_idx, 1] = chaser_y
        self.chaser_base_pos[envs_idx, 2] = self.chaser_init_pos[2]
        self.chaser_base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.chaser.set_pos(self.chaser_base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.chaser.set_quat(self.chaser_base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.chaser_base_lin_vel[envs_idx] = 0
        self.chaser_base_ang_vel[envs_idx] = 0
        self.chaser.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.runner_actions[envs_idx] = 0.0
        self.runner_last_actions[envs_idx] = 0.0
        self.chaser_actions[envs_idx] = 0.0
        self.chaser_last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.caught[envs_idx] = False
        self.survival_time[envs_idx] = 0.0

        # compute initial distance
        self.prev_distance[envs_idx] = self._compute_distance()[envs_idx]

        # initialize projected_gravity
        self.runner_projected_gravity[envs_idx] = self.global_gravity[envs_idx]
        self.chaser_projected_gravity[envs_idx] = self.global_gravity[envs_idx]

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
        # compute initial observations
        self._update_robot_state(self.runner, "runner")
        self._update_robot_state(self.chaser, "chaser")
        return self.obs_buf, self.extras


class TagGameRunnerEnv(TagGameEnv):
    """Environment for training the runner (逃げ側)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_speed = kwargs.get("max_speed", 1.5)

    def compute_chaser_commands(self, chaser_pos, runner_pos):
        """Chaser's pursuit algorithm - customize by editing this method"""
        direction = runner_pos[:, :2] - chaser_pos[:, :2]
        distance = torch.norm(direction, dim=-1, keepdim=True) + 1e-6
        direction = direction / distance
        commands = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        commands[:, 0] = direction[:, 0] * self.max_speed  # x velocity
        commands[:, 1] = direction[:, 1] * self.max_speed  # y velocity
        commands[:, 2] = 0.0  # angular velocity
        return commands

    def step(self, actions):
        """Advance the simulation by applying the actions of the runner"""
        # runner actions
        self.runner_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_runner_actions = self.runner_last_actions if self.simulate_action_latency else self.runner_actions
        target_runner_dof_pos = exec_runner_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.runner.control_dofs_position(target_runner_dof_pos, self.motors_dof_idx)

        # chaser: move by algorithm or policy
        if self.chaser_policy is not None:
            chaser_obs = self._compute_obs("chaser", self.chaser_actions)
            with torch.no_grad():
                chaser_actions = self.chaser_policy(chaser_obs)
        else:
            # Compute commands using algorithm
            chaser_commands = self.compute_chaser_commands(self.chaser_base_pos, self.runner_base_pos)
            self.commands[:] = chaser_commands
            # Reflect commands in chaser's observations to generate actions
            # Simply move forward
            chaser_actions = torch.zeros_like(self.chaser_actions)

        self.chaser_actions = torch.clip(chaser_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_chaser_actions = self.chaser_last_actions if self.simulate_action_latency else self.chaser_actions
        target_chaser_dof_pos = exec_chaser_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.chaser.control_dofs_position(target_chaser_dof_pos, self.motors_dof_idx)

        # simulation step
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.survival_time += self.dt

        self._update_robot_state(self.runner, "runner")
        self._update_robot_state(self.chaser, "chaser")

        # Check collision with walls
        self.runner_wall_collision = self._check_wall_collision(self.runner_base_pos)
        self.chaser_wall_collision = self._check_wall_collision(self.chaser_base_pos)

        # euler angles for termination check
        runner_base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.runner_base_quat) * self.inv_base_init_quat, self.runner_base_quat),
            rpy=True,
            degrees=True,
        )

        # Check capture
        distance = self._compute_distance()
        self.caught = distance < self.catch_distance

        # Update commands for runner (escape direction)
        escape_direction = self.runner_base_pos[:, :2] - self.chaser_base_pos[:, :2]
        escape_distance = torch.norm(escape_direction, dim=-1, keepdim=True) + 1e-6
        escape_direction = escape_direction / escape_distance
        self.commands[:, 0] = escape_direction[:, 0] * self.max_speed
        self.commands[:, 1] = escape_direction[:, 1] * self.max_speed
        self.commands[:, 2] = 0.0

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(runner_base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(runner_base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.caught

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

        # Update distance
        self.prev_distance[:] = distance

        # compute observations (runner)
        self.obs_buf = self._compute_obs("runner", self.runner_actions)

        self.runner_last_actions[:] = self.runner_actions[:]
        self.chaser_last_actions[:] = self.chaser_actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        obs, extras = super().reset()
        # compute observations for runner
        self.obs_buf = self._compute_obs("runner", self.runner_actions)
        return self.obs_buf, extras

    # ------------ reward functions for runner ----------------
    def _reward_survive(self):
        """Survival reward"""
        return torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_distance_from_chaser(self):
        """Reward based on distance from chaser"""
        distance = self._compute_distance()
        return distance

    def _reward_escape_velocity(self):
        """Reward for escape velocity"""
        escape_direction = self.runner_base_pos[:, :2] - self.chaser_base_pos[:, :2]
        escape_distance = torch.norm(escape_direction, dim=-1, keepdim=True) + 1e-6
        escape_direction = escape_direction / escape_distance
        velocity = self.runner_base_lin_vel[:, :2]
        escape_velocity = torch.sum(velocity * escape_direction, dim=-1)
        return torch.clamp(escape_velocity, min=0.0)

    def _reward_wall_collision(self):
        """Penalty for wall collision"""
        return self.runner_wall_collision.float()


class TagGameChaserEnv(TagGameEnv):
    """Environment for training the chaser (鬼側)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_speed = kwargs.get("max_speed", 1.5)

    def compute_runner_commands(self, runner_pos, chaser_pos):
        """Runner's escape algorithm - customize by editing this method"""
        direction = runner_pos[:, :2] - chaser_pos[:, :2]
        distance = torch.norm(direction, dim=-1, keepdim=True) + 1e-6
        direction = direction / distance
        commands = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        commands[:, 0] = direction[:, 0] * self.max_speed  # x velocity
        commands[:, 1] = direction[:, 1] * self.max_speed  # y velocity
        commands[:, 2] = 0.0  # angular velocity
        return commands

    def step(self, actions):
        """Advance the simulation by applying the actions of the chaser"""
        # chaser actions
        self.chaser_actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_chaser_actions = self.chaser_last_actions if self.simulate_action_latency else self.chaser_actions
        target_chaser_dof_pos = exec_chaser_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.chaser.control_dofs_position(target_chaser_dof_pos, self.motors_dof_idx)

        # runner: move by algorithm or policy
        if self.runner_policy is not None:
            runner_obs = self._compute_obs("runner", self.runner_actions)
            with torch.no_grad():
                runner_actions = self.runner_policy(runner_obs)
        else:
            # Compute commands using algorithm
            runner_commands = self.compute_runner_commands(self.runner_base_pos, self.chaser_base_pos)
            self.commands[:] = runner_commands
            # Simply move forward
            runner_actions = torch.zeros_like(self.runner_actions)

        self.runner_actions = torch.clip(runner_actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_runner_actions = self.runner_last_actions if self.simulate_action_latency else self.runner_actions
        target_runner_dof_pos = exec_runner_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.runner.control_dofs_position(target_runner_dof_pos, self.motors_dof_idx)

        # simulation step
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.survival_time += self.dt

        self._update_robot_state(self.runner, "runner")
        self._update_robot_state(self.chaser, "chaser")

        # Check collision with walls
        self.runner_wall_collision = self._check_wall_collision(self.runner_base_pos)
        self.chaser_wall_collision = self._check_wall_collision(self.chaser_base_pos)

        # euler angles for termination check
        chaser_base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.chaser_base_quat) * self.inv_base_init_quat, self.chaser_base_quat),
            rpy=True,
            degrees=True,
        )

        # Check capture
        distance = self._compute_distance()
        self.caught = distance < self.catch_distance

        # Update commands for chaser (chase direction)
        chase_direction = self.runner_base_pos[:, :2] - self.chaser_base_pos[:, :2]
        chase_distance = torch.norm(chase_direction, dim=-1, keepdim=True) + 1e-6
        chase_direction = chase_direction / chase_distance
        self.commands[:, 0] = chase_direction[:, 0] * self.max_speed
        self.commands[:, 1] = chase_direction[:, 1] * self.max_speed
        self.commands[:, 2] = 0.0

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(chaser_base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(chaser_base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.caught  # End episode if caught

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

        # Update distance
        self.prev_distance[:] = distance

        # compute observations (chaser)
        self.obs_buf = self._compute_obs("chaser", self.chaser_actions)

        self.runner_last_actions[:] = self.runner_actions[:]
        self.chaser_last_actions[:] = self.chaser_actions[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        obs, extras = super().reset()
        # compute observations for chaser
        self.obs_buf = self._compute_obs("chaser", self.chaser_actions)
        return self.obs_buf, extras

    # ------------ reward functions for chaser ----------------
    def _reward_catch(self):
        """Catch reward"""
        return self.caught.float()

    def _reward_approach(self):
        """Approach reward"""
        distance = self._compute_distance()
        approach = self.prev_distance - distance
        return torch.clamp(approach, min=0.0)

    def _reward_chase_velocity(self):
        """Reward for chase velocity"""
        chase_direction = self.runner_base_pos[:, :2] - self.chaser_base_pos[:, :2]
        chase_distance = torch.norm(chase_direction, dim=-1, keepdim=True) + 1e-6
        chase_direction = chase_direction / chase_distance
        velocity = self.chaser_base_lin_vel[:, :2]
        chase_velocity = torch.sum(velocity * chase_direction, dim=-1)
        return torch.clamp(chase_velocity, min=0.0)

    def _reward_wall_collision(self):
        """Penalty for wall collision"""
        return self.chaser_wall_collision.float()
