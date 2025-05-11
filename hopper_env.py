import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import mujoco.viewer

class HopperBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path("2d_hopper.xml")
        self.data = mujoco.MjData(self.model)

        self.x_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_x')
        self.z_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_z')

        self.action_space = spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32)
        high = np.array([np.inf]*8, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.viewer = None
        self.sim_steps_per_step = 10

        # --- Data logging additions ---
        self.trajectory_log = []
        self.last_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        self.last_obs = obs.copy()  # store initial obs
        self.trajectory_log = []    # clear log at reset
        return obs, {}

    def step(self, action):
        torque = np.clip(action, -200, 200)
        self.data.ctrl[0] = torque

        for _ in range(self.sim_steps_per_step):
            mujoco.mj_step(self.model, self.data)

            # Wrap-around logic
            x_pos = self.data.qpos[self.x_joint_id]
            if x_pos > 5.0:
                self.data.qpos[self.x_joint_id] = -5.0
                self.data.qpos[self.z_joint_id] = 0.2
                if abs(self.data.qvel[self.x_joint_id]) < 1e-3:
                    self.data.qvel[self.x_joint_id] = 1.0
                mujoco.mj_forward(self.model, self.data)
            elif x_pos < -5.0:
                self.data.qpos[self.x_joint_id] = 5.0
                self.data.qpos[self.z_joint_id] = 0.2
                if abs(self.data.qvel[self.x_joint_id]) < 1e-3:
                    self.data.qvel[self.x_joint_id] = -1.0
                mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()

        # Reward computation
        theta_base = obs[2]
        theta_hip  = obs[3]
        theta_torso = theta_base + theta_hip
        hip_penalty = 0.1 * abs(theta_hip)

        upright_bonus = np.cos(theta_torso)
        alive_bonus = 1.0

        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        z_torso = self.data.xpos[torso_id][2]
        reward = upright_bonus + alive_bonus + 10 * (max(0, z_torso - 0.4))**2 - hip_penalty

        done = abs(theta_torso) > np.pi/2 or z_torso < 0.35

        # --- Log transition ---
        if self.last_obs is not None:
            x_t = self.last_obs
            u_t = np.array(action).flatten()
            x_next = obs.copy()
            self.trajectory_log.append((x_t, u_t, x_next))
        self.last_obs = obs.copy()

        return obs, reward, done, False, {}

    def _get_obs(self):
        x = self.data.qpos[0]
        z = self.data.qpos[1]
        theta = self.data.qpos[2]
        x_dot = self.data.qvel[0]
        z_dot = self.data.qvel[1]
        theta_dot = self.data.qvel[2]
        hip_angle = self.data.qpos[3]
        hip_vel = self.data.qvel[3]
        return np.array([x, z, theta, x_dot, z_dot, theta_dot, hip_angle, hip_vel], dtype=np.float32)

    def get_logged_data(self):
        """Return list of (x, u, x_next) triplets."""
        return self.trajectory_log

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
