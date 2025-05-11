import numpy as np
from stable_baselines3 import PPO
from hopper_env import HopperBalanceEnv

# --- Setup ---
env = HopperBalanceEnv(render_mode=None)
model = PPO.load("./logs/best_model/best_model.zip")
timesteps = 2000  # Number of total steps to collect

obs, _ = env.reset()
done = False
steps = 0

# --- Rollout loop ---
while steps < timesteps:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    steps += 1
    if done:
        obs, _ = env.reset()

# --- Extract logged data ---
log = env.get_logged_data()
x = np.array([item[0] for item in log])
u = np.array([item[1] for item in log])
x_next = np.array([item[2] for item in log])

# --- Compute xdot_true ---
# dt = sim_steps_per_step * mujoco timestep
dt = env.sim_steps_per_step * env.model.opt.timestep
xdot_true = (x_next - x) / dt

# --- Save to file ---
np.savez("residual_data.npz", x=x, u=u, x_next=x_next, xdot_true=xdot_true, dt=dt)
print(f"Saved {len(x)} transitions to residual_data.npz")
