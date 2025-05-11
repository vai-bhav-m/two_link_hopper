# import mujoco
# import mujoco.viewer
# import numpy as np
# import time

# # Load model and data
# model = mujoco.MjModel.from_xml_path("2d_hopper.xml")
# data = mujoco.MjData(model)

# # Get joint indices
# x_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_x')
# z_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_z')

# print("Simulation running with automated torque control...")

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     t = 0.0
#     while viewer.is_running():
#         step_start_time = time.time()

#         # Apply a sinusoidal torque
#         data.ctrl[0] = 10 * np.sin(t)

#         # Step the simulation
#         mujoco.mj_step(model, data)

#         # Get current x position
#         x_pos = data.qpos[x_joint_id]

#         # --- Wrap-around logic ---
#         if x_pos > 5.0:
#             data.qpos[x_joint_id] = -5.0
#             data.qpos[z_joint_id] = 0.2  # lift off ground to avoid contact
#             if abs(data.qvel[x_joint_id]) < 1e-3:  # optional: ensure some velocity
#                 data.qvel[x_joint_id] = 1.0
#             mujoco.mj_forward(model, data)  # reinitialize physics

#         elif x_pos < -5.0:
#             data.qpos[x_joint_id] = 5.0
#             data.qpos[z_joint_id] = 0.2
#             if abs(data.qvel[x_joint_id]) < 1e-3:
#                 data.qvel[x_joint_id] = -1.0
#             mujoco.mj_forward(model, data)

#         # Sync viewer
#         viewer.sync()

#         # --- Real-time synchronization ---
#         elapsed = time.time() - step_start_time
#         sim_dt = model.opt.timestep
#         if elapsed < sim_dt:
#             time.sleep(sim_dt - elapsed)

#         t += sim_dt

import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model and data
model = mujoco.MjModel.from_xml_path("2d_hopper.xml")
data = mujoco.MjData(model)

# Get joint indices
x_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_x')
z_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'slide_z')

print("Simulation running with real-time physics sync...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_wall_time = time.perf_counter()

    while viewer.is_running():
        current_wall_time = time.perf_counter()
        wall_elapsed = current_wall_time - start_wall_time

        # Catch up simulation if behind real-time
        while data.time < wall_elapsed:
            # Apply control
            data.ctrl[0] = 100 * np.sin(data.time)

            mujoco.mj_step(model, data)

            # --- Wrap-around logic ---
            x_pos = data.qpos[x_joint_id]
            if x_pos > 5.0:
                data.qpos[x_joint_id] = -5.0
                data.qpos[z_joint_id] = 0.2  # lift off ground
                if abs(data.qvel[x_joint_id]) < 1e-3:
                    data.qvel[x_joint_id] = 1.0
                mujoco.mj_forward(model, data)
            elif x_pos < -5.0:
                data.qpos[x_joint_id] = 5.0
                data.qpos[z_joint_id] = 0.2
                if abs(data.qvel[x_joint_id]) < 1e-3:
                    data.qvel[x_joint_id] = -1.0
                mujoco.mj_forward(model, data)

        # Always sync viewer
        viewer.sync()

        # Optional debug print
        # print(f"Sim time: {data.time:.3f}, Wall time: {wall_elapsed:.3f}")
