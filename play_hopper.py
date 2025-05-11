# from hopper_env import HopperBalanceEnv
# from stable_baselines3 import PPO
# import time

# env = HopperBalanceEnv(render_mode="human")
# model = PPO.load("hopper_ppo_model_60_10")

# obs, _ = env.reset()
# tstep = 0
# while True:
#     tstep += 1
#     action, _ = model.predict(obs)
#     obs, reward, done, _, _ = env.step(action)
#     env.render()
#     time.sleep(0.01)
#     if done:
#         # break
#         print(f"Was alive for {tstep}")
#         tstep = 0
#         obs, _ = env.reset()


from hopper_env import HopperBalanceEnv
from stable_baselines3 import PPO
import time

env = HopperBalanceEnv(render_mode="human")
model = PPO.load("./logs/best_model/best_model.zip")

num_episodes = 0
total_timesteps = 0
episode_rewards = []
episode_lengths = []

start_time = time.time()

obs, _ = env.reset()
tstep = 0
ep_reward = 0

while True:
    tstep += 1
    total_timesteps += 1

    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, _, _ = env.step(action)
    ep_reward += reward
    env.render()
    time.sleep(0.01)

    if done:
        num_episodes += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(tstep)

        if num_episodes % 10 == 0:
            time_elapsed = time.time() - start_time
            fps = int(total_timesteps / time_elapsed)
            ep_rew_mean = sum(episode_rewards[-10:]) / 10
            ep_len_mean = sum(episode_lengths[-10:]) / 10

            print("="*40)
            print(f"rollout/")
            print(f"  ep_len_mean     {ep_len_mean:.0f}")
            print(f"  ep_rew_mean     {ep_rew_mean:.0f}")
            print(f"time/")
            print(f"  fps             {fps}")
            print(f"  episodes        {num_episodes}")
            print(f"  total_timesteps {total_timesteps}")
            print("="*40)

        # Reset episode
        tstep = 0
        ep_reward = 0
        obs, _ = env.reset()
