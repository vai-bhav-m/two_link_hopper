from hopper_env import HopperBalanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from video_callback import VideoCallback
import os


# Create training and evaluation environments
train_env = HopperBalanceEnv(render_mode=None)
eval_env = HopperBalanceEnv(render_mode=None)

test_str = "./gifs_free_hip_pen"
os.makedirs(test_str, exist_ok=True)


video_callback = VideoCallback(eval_env, every=5000, gif_dir=test_str)


# Setup evaluation callback: evaluate every 5000 steps
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/eval_log/",
    eval_freq=5000,
    deterministic=True,
    render=False
)

# Create PPO model with TensorBoard logging
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_hopper_tensorboard/"
)

# Train with evaluation callback
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, video_callback]
)

# Save final model
model.save("hopper_ppo_model")

train_env.close()
eval_env.close()