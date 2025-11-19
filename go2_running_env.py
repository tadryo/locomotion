import torch
from go2_env import Go2Env


class Go2RunningEnv(Go2Env):
    """Go2ロボットの走行動作に特化した環境クラス"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)

        # 走行用の追加バッファ
        self.last_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.feet_contact_forces = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)

    def step(self, actions):
        # 前回の位置を保存（forward distance計算用）
        self.last_base_pos[:] = self.base_pos.clone()

        # 親クラスのstepを実行
        obs, rew, done, extras = super().step(actions)

        return obs, rew, done, extras

    def reset_idx(self, envs_idx):
        # 親クラスのresetを実行
        super().reset_idx(envs_idx)

        # 走行用バッファのリセット
        if len(envs_idx) > 0:
            self.last_base_pos[envs_idx] = self.base_pos[envs_idx].clone()

    # ------------ カスタム報酬関数 ----------------

    def _reward_forward_distance(self):
        """前進距離を報酬化

        ロボットが実際に前進した距離（X軸方向）を直接報酬として与える。
        速度ではなく距離を報酬化することで、より確実な前進を促す。
        """
        forward_distance = self.base_pos[:, 0] - self.last_base_pos[:, 0]
        # 負の値（後退）はペナルティとして扱う
        return torch.clamp(forward_distance, min=0.0)

    def _reward_diagonal_gait(self):
        """対角歩容（トロット）を奨励

        走行時の効率的な歩容パターン。
        対角線上の脚（FR-RL、FL-RR）が同期して動くことを奨励。
        脚のインデックス:
        - FR (Front Right): 0-2
        - FL (Front Left): 3-5
        - RR (Rear Right): 6-8
        - RL (Rear Left): 9-11
        """
        # 対角線上の脚ペアの関節速度の相関を計算
        # FR と RL の股関節・大腿関節の速度
        fr_vel = self.dof_vel[:, [0, 1]]  # FR hip, thigh
        rl_vel = self.dof_vel[:, [9, 10]]  # RL hip, thigh

        # FL と RR の股関節・大腿関節の速度
        fl_vel = self.dof_vel[:, [3, 4]]  # FL hip, thigh
        rr_vel = self.dof_vel[:, [6, 7]]  # RR hip, thigh

        # 対角線ペアの速度パターンの類似性を計算
        # 符号が逆であることを期待（一方が伸びているとき、もう一方は縮んでいる）
        diagonal1_correlation = -torch.sum(fr_vel * rl_vel, dim=1)
        diagonal2_correlation = -torch.sum(fl_vel * rr_vel, dim=1)

        # 両対角線の平均をとる
        diagonal_gait_reward = (diagonal1_correlation + diagonal2_correlation) / 2.0

        return torch.clamp(diagonal_gait_reward, min=0.0)

    def _reward_aligned_hips(self):
        """ヒップ関節の整列を奨励

        4つのヒップ関節が同じ角度を保つことで、
        ロボットの姿勢の安定性と対称性を維持する。
        """
        # ヒップ関節のインデックス: 0(FR), 3(FL), 6(RR), 9(RL)
        hip_positions = self.dof_pos[:, [0, 3, 6, 9]]

        # ヒップ関節角度の標準偏差を計算（小さいほど整列している）
        hip_std = torch.std(hip_positions, dim=1)

        # 標準偏差が小さいほど高い報酬
        return torch.exp(-hip_std * 10.0)

    def _reward_straight_line(self):
        """直進性を報酬化

        Y軸方向の移動を抑制し、X軸方向への直進を奨励する。
        ロボットが横にずれないようにする。
        """
        # Y軸方向の速度のペナルティ
        lateral_velocity = torch.abs(self.base_lin_vel[:, 1])

        # Y軸方向の位置の偏差もペナルティ
        lateral_deviation = torch.abs(self.base_pos[:, 1])

        # 両方を組み合わせて、直進性を評価
        straight_reward = torch.exp(-(lateral_velocity + lateral_deviation * 0.5))

        return straight_reward

    def _reward_foot_clearance(self):
        """足の適切な持ち上げを奨励

        走行時には足を地面から適切に持ち上げる必要がある。
        膝関節の角速度が大きいことを奨励することで、
        ダイナミックな足の動きを促す。
        """
        # 膝（calf）関節のインデックス: 2(FR), 5(FL), 8(RR), 11(RL)
        calf_velocities = torch.abs(self.dof_vel[:, [2, 5, 8, 11]])

        # 膝関節の平均角速度
        avg_calf_velocity = torch.mean(calf_velocities, dim=1)

        # 適度な速度域を報酬化（過度に速すぎるのは不安定）
        optimal_velocity = 3.0
        foot_clearance_reward = torch.exp(-torch.abs(avg_calf_velocity - optimal_velocity) / 2.0)

        return foot_clearance_reward

    def _reward_energy_efficiency(self):
        """エネルギー効率を報酬化

        トルクの二乗和を最小化することで、
        エネルギー効率の良い動きを奨励する。
        """
        # アクションの大きさ（トルクに比例）の二乗和
        torque_penalty = torch.sum(torch.square(self.actions), dim=1)

        return torque_penalty
