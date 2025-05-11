from stable_baselines3.common.callbacks import BaseCallback
import imageio
import mujoco  # official mujoco
import numpy as np

class VideoCallback(BaseCallback):
    def __init__(self, env, every=10000, gif_dir="./gifs/", max_frames=500, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.every = every
        self.gif_dir = gif_dir
        self.max_frames = max_frames
        self.renderer = mujoco.Renderer(self.env.model)
        self.cam = mujoco.MjvCamera()

    def _on_step(self):
        if self.n_calls % self.every == 0:
            frames = []
            obs, _ = self.env.reset()
            done = False
            steps = 0

            while not done and steps < self.max_frames:
                # Set camera properties
                self.cam.lookat[:] = [self.env.data.qpos[0], 0, 0.3]
                self.cam.distance = 1

                self.renderer.update_scene(self.env.data, camera=self.cam)
                img = self.renderer.render()
                frames.append(img)

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                steps += 1

            filename = f"{self.gif_dir}/video_{self.n_calls}.gif"
            imageio.mimsave(filename, frames, duration=0.02)
            print(f"Saved GIF to {filename}")

        return True
